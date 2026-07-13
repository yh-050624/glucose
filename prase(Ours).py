import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# ================= Configuration =================
CONFIG = {
    'INPUT_DIR': 'data/Ohio T1DM',
    'OUTPUT_DIR': 'data/CSDI',
    'PATIENT_IDS': ['563', '570', '591'],
    'TRAIN_TEMPLATE': '{pid}-ws-training.xml',
    'TEST_TEMPLATE': '{pid}-ws-testing.xml',
    'STEP_MIN': 5,
    'INSULIN_DURATION_MIN': 240,
    'CARB_DURATION_MIN': 180
}


class OhioProposedProcessor:
    """
    Proposed Method: Raw data preservation with masking.

    Processing Pipeline:
    1. Parse XML data from Ohio T1DM dataset
    2. Resample to fixed temporal grid (introduces NaN naturally)
    3. NO imputation for CGM values - preserve NaN
    4. Zero-fill for insulin and carbohydrate inputs (impulse signals)
    5. Generate binary mask: cbg_mask (1=observed, 0=missing)
    6. Compute physiological features (IOB, COB)

    Output Format:
    - cbg: Original values with NaN for missing data
    - cbg_mask: Binary indicator (1=observed, 0=missing)
    - basal, bolus, carbInput: Zero-filled
    - IOB, COB: Physiological features

    Attributes:
        step_min (int): Temporal resolution in minutes
        insulin_duration (int): Insulin action duration in minutes
        carb_duration (int): Carbohydrate absorption duration in minutes
    """

    def __init__(self, step_min: int = 5,
                 insulin_duration_min: int = 240,
                 carb_duration_min: int = 180):
        """
        Initialize processor.

        Args:
            step_min (int): Temporal resolution in minutes
            insulin_duration_min (int): Insulin action duration in minutes
            carb_duration_min (int): Carb absorption duration in minutes
        """
        self.step_min = step_min
        self.insulin_duration = insulin_duration_min
        self.carb_duration = carb_duration_min

    def run_pipeline(self, xml_path: str, output_csv_path: str) -> None:
        """Execute complete preprocessing pipeline."""
        if not os.path.exists(xml_path):
            print(f"[Skip] File not found: {xml_path}")
            return

        print(f"[*] Processing: {os.path.basename(xml_path)}")

        # Step 1: Parse XML
        raw_df = self._parse_xml(xml_path)
        if raw_df.empty:
            return

        # Step 2: Resample to temporal grid (introduces NaN)
        grid_df = self._resample_to_grid(raw_df)

        # Step 3: Zero-fill impulse data only
        for col in ['basal', 'bolus', 'carbInput']:
            if col not in grid_df.columns:
                grid_df[col] = 0
            grid_df[col] = grid_df[col].fillna(0)

        # Step 4: Generate mask BEFORE any imputation
        # CRITICAL: This must come before any interpolation
        grid_df['cbg_mask'] = (~grid_df['cbg'].isna()).astype(int)

        missing_count = (grid_df['cbg_mask'] == 0).sum()
        observed_count = (grid_df['cbg_mask'] == 1).sum()
        missing_rate = missing_count / len(grid_df) * 100

        print(f"    -> CBG: {observed_count} observed, {missing_count} missing ({missing_rate:.1f}%)")

        # Step 5: Compute physiological features
        # Note: IOB/COB computed on complete impulse data (no missing values)
        grid_df = self._compute_pharmacokinetics(grid_df)

        # Step 6: Save to CSV
        # IMPORTANT: cbg column contains NaN - this is intentional!
        grid_df.to_csv(output_csv_path, index_label='timestamp')
        print(f"    -> Saved: {output_csv_path} (n={len(grid_df)} records)")
        print(f"       NOTE: cbg column contains {missing_count} NaN values (preserved for model)")

    def _compute_pharmacokinetics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute physiological features: IOB and COB.

        Uses same linear decay models as Baseline 2 for fair comparison.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with IOB and COB columns added
        """
        # === IOB Calculation ===
        steps_insulin = int(self.insulin_duration / self.step_min)
        bolus = df['bolus'].values
        iob = np.zeros(len(df))
        decay_insulin = np.linspace(1, 0, steps_insulin)

        for i in range(len(bolus)):
            if bolus[i] > 0:
                end_idx = min(i + steps_insulin, len(df))
                iob[i:end_idx] += bolus[i] * decay_insulin[:(end_idx - i)]

        df['IOB'] = iob

        # === COB Calculation ===
        steps_carb = int(self.carb_duration / self.step_min)
        carbs = df['carbInput'].values
        cob = np.zeros(len(df))
        decay_carb = np.linspace(1, 0, steps_carb)

        for i in range(len(carbs)):
            if carbs[i] > 0:
                end_idx = min(i + steps_carb, len(df))
                cob[i:end_idx] += carbs[i] * decay_carb[:(end_idx - i)]

        df['COB'] = cob

        return df

    def _parse_xml(self, xml_path: str) -> pd.DataFrame:
        """Parse Ohio T1DM XML file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"    [Error] XML parsing failed: {e}")
            return pd.DataFrame()

        data = []

        for child in root:
            tag = child.tag.lower()

            # Map XML tags to column names
            if 'glucose' in tag:
                col_name = 'cbg'
                ts_attr = 'ts'
                val_attr = 'value'
            elif tag == 'basal':
                col_name = 'basal'
                ts_attr = 'ts'
                val_attr = 'value'
            elif tag == 'bolus':
                col_name = 'bolus'
                ts_attr = 'ts_begin'
                val_attr = 'dose'
            elif tag == 'meal':
                col_name = 'carbInput'
                ts_attr = 'ts'
                val_attr = 'carbs'
            else:
                continue

            for event in child:
                ts_raw = event.get(ts_attr)
                val_raw = event.get(val_attr)

                if ts_raw and val_raw:
                    try:
                        timestamp = pd.to_datetime(ts_raw, dayfirst=True, errors='coerce')
                        if pd.notna(timestamp):
                            data.append({
                                'timestamp': timestamp,
                                'type': col_name,
                                'value': float(val_raw)
                            })
                    except (ValueError, TypeError):
                        continue

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        print(f"    -> Extracted {len(df)} raw records")

        df = df.drop_duplicates(subset=['timestamp', 'type'], keep='first')
        df_pivot = df.pivot_table(index='timestamp', columns='type',
                                  values='value', aggfunc='mean')

        for col in ['cbg', 'basal', 'bolus', 'carbInput']:
            if col not in df_pivot.columns:
                df_pivot[col] = np.nan

        return df_pivot

    def _resample_to_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to fixed temporal grid.

        Key difference from baselines: This naturally introduces NaN
        for missing time points, which is the desired behavior.
        """
        if df.empty:
            return df

        full_index = pd.date_range(start=df.index.min(), end=df.index.max(),
                                   freq=f'{self.step_min}min')

        tolerance = pd.Timedelta(minutes=self.step_min / 2)
        resampled_df = pd.DataFrame(index=full_index)

        for col in df.columns:
            resampled_df[col] = df[col].reindex(full_index, method='nearest',
                                                tolerance=tolerance)

        print(f"    -> Resampled to {len(resampled_df)} time points")

        return resampled_df


def main():
    """Main execution function."""
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

    processor = OhioProposedProcessor(
        step_min=CONFIG['STEP_MIN'],
        insulin_duration_min=CONFIG['INSULIN_DURATION_MIN'],
        carb_duration_min=CONFIG['CARB_DURATION_MIN']
    )

    print("=" * 70)
    print("Ohio T1DM - Proposed Method (Raw Data with Masking)")
    print("=" * 70)
    print("NOTE: This method preserves NaN values in CBG column")
    print("      for downstream imputation/prediction models.")
    print("=" * 70)

    for pid in CONFIG['PATIENT_IDS']:
        print(f"\n{'=' * 70}")
        print(f"Patient ID: {pid}")
        print(f"{'=' * 70}")

        xml_train = os.path.join(CONFIG['INPUT_DIR'], CONFIG['TRAIN_TEMPLATE'].format(pid=pid))
        csv_train = os.path.join(CONFIG['OUTPUT_DIR'], f"{pid}_training_raw.csv")
        processor.run_pipeline(xml_train, csv_train)

        xml_test = os.path.join(CONFIG['INPUT_DIR'], CONFIG['TEST_TEMPLATE'].format(pid=pid))
        csv_test = os.path.join(CONFIG['OUTPUT_DIR'], f"{pid}_testing_raw.csv")
        processor.run_pipeline(xml_test, csv_test)

    print(f"\n{'=' * 70}")
    print("[Done] Proposed method preprocessing completed.")
    print(f"Output directory: {CONFIG['OUTPUT_DIR']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
