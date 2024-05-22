import numpy as np
import polars as pl
from sklearn.preprocessing import OrdinalEncoder
from tqdm.notebook import tqdm

from dataset_init_params import date_cols, ignored_features, education_grades, agg_col_params
from lf_processing import scan_file, scan_files, handle_dates


class Dataset:
    """
    Train/test dataset generator from all .parquet table files.
    Feature selection included.

    Examples
    --------
    >>> ds = Dataset()
    >>> df_train = ds.get_dataframe()
    >>> df_test = ds.get_dataframe(train=False)
    """
    def __init__(self) -> None:
        self.date_cols = date_cols()
        self.tax_filenames = [
            'tax_registry_a_1', 'tax_registry_b_1', 'tax_registry_c_1'
        ]
        self.train_filename_prefix = 'parquet_files/train/train_'
        self.test_filename_prefix = 'parquet_files/test/test_'
        self.ignored_features = ignored_features()
        self.education_grades = education_grades()
        self.marital_ord_enc = self._get_marital_ord_enc()
        self.agg_col_params = agg_col_params()

    def _get_ignored_features(self, cols: list, table_name: str) -> list:
        """
        Get ignored features to drop them.

        Parameters
        ----------
        cols: list like
            columns
        table_name: str
            key to find in self.ignored_features

        Returns
        -------
        ans: list
        """
        ign_features = self.ignored_features[table_name]
        ans = []
        for col in cols:
            for sub in ign_features:
                if sub in col:
                    ans.append(col)
                    break
        return ans

    def _get_marital_ord_enc(self) -> OrdinalEncoder:
        """
        Generate OrdinalEncoder for maritalst_385M feature.

        Returns
        -------
        out: OrdinalEncoder
        """
        lf = scan_file(self.train_filename_prefix + 'static_cb_0.parquet')
        lf = lf.select('maritalst_385M').cast(pl.String).unique()
        df_enc = lf.collect()
        uniq_values = np.unique(np.array(df_enc).ravel().astype(str))
        marital_ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value',
                                         unknown_value=-1,
                                         dtype=int)
        marital_ord_enc = marital_ord_enc.fit(uniq_values.reshape(-1, 1))
        return marital_ord_enc

    def _aggregate_columns(self, lf: pl.LazyFrame,
                           table_name: str) -> pl.LazyFrame:
        """
        Агрегирует колонки в соответствии с настройками agg_col_params.

        Parameters
        ----------
        lf: pl.LazyFrame

        table_name: str
            To select settings

        Returns
        -------
        lf: pl.LazyFrame

        """
        if table_name in self.agg_col_params:
            agg_col = self.agg_col_params[table_name]
        else:
            return lf

        for alias, params in agg_col.items():
            is_min = False
            if len(params) == 2:
                sub, dtype = params
            else:
                sub, dtype, is_min = params
            if type(sub) == str:
                cols = [col for col in lf.columns if sub in col]
            elif type(sub) == list:
                cols = sub
            elif type(sub) == dict:
                cols = []
                for k in sub.keys():
                    k_cols = [col for col in lf.columns if k in col]
                    cols.extend(k_cols)
                for v in sub.values():
                    if type(v) == str:
                        cols.append(v)
                    elif type(v) == list:
                        cols.extend(v)
                    else:
                        continue
            else:
                continue
            if is_min:
                lf = lf.with_columns(
                    pl.min_horizontal(cols).alias(alias).cast(dtype))
            else:
                lf = lf.with_columns(
                    pl.max_horizontal(cols).alias(alias).cast(dtype))
            lf = lf.drop(cols)
        return lf

    def generate_data(self, train: bool = True) -> pl.LazyFrame:
        """
        Generate train or test data reading .parquet files.
        1. Read base table.
        2. Read rest tables and left join it to base.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        lf = self._base_table(train)
        tables_id = {
            '_static_cb': self._static_cb,
            '_static_0': self._static_0,
            '_applprev_1': self._applprev_1,
            '_tax_registry': self._tax_registry,
            '_cb_a_1': self._cb_a_1,
            '_cb_b_1': self._cb_b_1,
            '_person_1': self._person_1,
            '_applprev_2': self._applprev_2,
            '_person_2': self._person_2,
            '_cb_a_2': self._cb_a_2,
        }
        for table_id, func in tables_id.items():
            lf = lf.join(func(train),
                         how="left",
                         on="case_id",
                         suffix=table_id)
        table_id = 'main'
        cols = self._get_ignored_features(lf.columns, table_id)
        lf = lf.drop(cols)
        lf = lf.pipe(handle_dates)
        lf = self._aggregate_columns(lf, table_id)
        lf = lf.with_columns(
            pl.col('opencred_647L').cast(pl.Int8),
            pl.col('remitter_829L').cast(pl.Int8),
            pl.col('isbidproduct_390L').cast(pl.Int8),
            pl.col('isbidproduct').cast(pl.Int8),
            pl.col('safeguarantyflag_411L').cast(pl.Int8),
            pl.col('contaddr_smempladdr').cast(pl.Int8),
        )
        return lf

    def get_dataframe(self,
                      train: bool = True,
                      indices_or_sections: int = 10) -> pl.DataFrame:
        """
        Собрать DataFrame из LazyFrame по n_chunks колонок.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)
        indices_or_sections: int = 10

        Returns
        -------
        df: pl.DataFrame

        """
        lf = self.generate_data(train)
        cols_split = np.array_split(lf.columns, indices_or_sections)
        df = lf.select(cols_split[0]).collect()
        for i in tqdm(range(1, len(cols_split))):
            df2 = lf.select(cols_split[i]).collect()
            df = pl.concat([df, df2], how='horizontal')
        return df

    # region TABLES
    def _base_table(self, train: bool = True) -> pl.LazyFrame:
        """
        Read and process base table.
        Create new features as month and weekday of decision date.
        Drop MONTH feature as non-informative.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        filename_prefix = self.train_filename_prefix if train else self.test_filename_prefix
        lf = scan_file(filename_prefix + 'base.parquet')
        lf = lf.with_columns(
            pl.col('date_decision').cast(pl.Date),
            pl.col('WEEK_NUM').cast(pl.Int32))
        lf = lf.with_columns(
            month_decision=pl.col('date_decision').dt.month(),
            weekday_decision=pl.col('date_decision').dt.weekday())
        lf = lf.drop('MONTH')
        return lf

    def _static_cb(self, train: bool = True) -> pl.LazyFrame:
        """
        Read and process _static_cb_0 data.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        filename_prefix = self.train_filename_prefix if train else self.test_filename_prefix
        lf = scan_file(filename_prefix + 'static_cb_0.parquet')
        cols = self._get_ignored_features(lf.columns, 'static_cb_0')
        lf = lf.drop(cols)
        # Days**.
        days_cols = [col for col in lf.columns if 'days' in col]
        lf = lf.with_columns(pl.col(days_cols).cast(pl.Int32))
        # Education.
        edu_cols = [col for col in lf.columns if 'education' in col]
        lf = lf.with_columns(
            pl.col(edu_cols).map_elements(lambda x: self.education_grades[x] if
            x in self.education_grades else 0, return_dtype=pl.Int32))
        lf = lf.with_columns(pl.col(edu_cols))
        lf = lf.with_columns(pl.max_horizontal(edu_cols).alias('education'))
        lf = lf.drop(edu_cols)
        # maritalst
        lf = lf.with_columns(
            pl.col('maritalst_385M').map_batches(
                lambda x: self.marital_ord_enc.transform(x.to_numpy().reshape(
                    -1, 1)).ravel().T))
        # numberofqueries_373L, pmtscount_423L
        lf = lf.with_columns(
            pl.col('numberofqueries_373L', 'pmtscount_423L').cast(pl.Int32))
        lf = lf.with_columns(pl.col('requesttype_4525192L'))
        lf = self._aggregate_columns(lf, 'static_cb')
        return lf

    def _static_0(self, train: bool = True) -> pl.LazyFrame:
        """
        Read and process _static_0_* data.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        filename_prefix = self.train_filename_prefix if train else self.test_filename_prefix
        lf = scan_files(filename_prefix + 'static_0_*.parquet')
        cols = self._get_ignored_features(lf.columns, 'static_0')
        lf = lf.drop(cols)
        # fill_null by 0 for int32 dtypes
        lf = lf.with_columns(
            pl.col('amtinstpaidbefduel24m_4187115A', 'cntpmts24_3658933L',
                   'daysoverduetolerancedd_3976961L').cast(pl.Int32))
        lf = self._aggregate_columns(lf, 'static_0')
        return lf

    def _applprev_1(self, train: bool = True) -> pl.LazyFrame:
        """
        Read and process _applprev_1_* data.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        filename_prefix = self.train_filename_prefix if train else self.test_filename_prefix
        table_id = 'applprev_1_*'
        lf = scan_files(filename_prefix + table_id + '.parquet',
                        depth=1,
                        date_col=self.date_cols[table_id])
        cols = self._get_ignored_features(lf.columns, table_id)
        lf = lf.drop(cols)
        # Education.
        edu_cols = [col for col in lf.columns if 'education' in col]
        lf = lf.with_columns(
            pl.col(edu_cols).map_elements(lambda x: self.education_grades[
                x] if x in self.education_grades else 0, return_dtype=pl.Int32).alias('education_1'))
        lf = lf.drop(edu_cols)
        lf = self._aggregate_columns(lf, table_id)
        return lf

    def _tax_registry(self, train: bool = True) -> pl.LazyFrame:
        """
        Read tax_registry data.
        Collect amount, empoyer_name, record_date columns to one.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        filename_prefix = self.train_filename_prefix if train else self.test_filename_prefix
        frames = []
        for tax_name in self.tax_filenames:
            lf = scan_file(filename_prefix + tax_name + '.parquet',
                           depth=1,
                           date_col=self.date_cols[tax_name])
            frames.append(lf)
        lf = frames[0].join(frames[1],
                            how="left",
                            on="case_id",
                            suffix=f"_tax_{0}")
        for i in range(1, len(frames) - 1):
            lf = lf.join(frames[i + 1],
                         how="left",
                         on="case_id",
                         suffix=f"_tax_{i}")
        amount_cols = [col for col in lf.columns if 'amount' in col]
        emp_name_cols = [col for col in lf.columns if 'name' in col]
        record_date_cols = [col for col in lf.columns if 'date' in col]
        group_cols = [col for col in lf.columns if 'num_group' in col]
        lf = lf.with_columns(
            pl.max_horizontal(amount_cols).alias('tax_amount'),
            pl.max_horizontal(emp_name_cols).alias('employer_name'),
            pl.max_horizontal(record_date_cols).alias('record_date'),
            pl.max_horizontal(group_cols).alias('num_group1_tax'))
        amount_cols.extend(emp_name_cols)
        amount_cols.extend(record_date_cols)
        amount_cols.extend(group_cols)
        lf = lf.drop(amount_cols)
        return lf

    def _cb_a_1(self, train: bool = True) -> pl.LazyFrame:
        """
        Read and process credit_bureau_a_1_* data.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        filename_prefix = self.train_filename_prefix if train else self.test_filename_prefix
        file_id = 'credit_bureau_a_1_*'
        lf = scan_files(filename_prefix + file_id + '.parquet',
                        depth=1,
                        date_col=self.date_cols[file_id])
        cols = self._get_ignored_features(lf.columns, file_id)
        lf = lf.drop(cols)
        lf = self._aggregate_columns(lf, file_id)
        return lf

    def _cb_a_2(self, train: bool = True) -> pl.LazyFrame:
        """
        Read and process credit_bureau_a_2_* data.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        filename_prefix = self.train_filename_prefix if train else self.test_filename_prefix
        file_id = 'credit_bureau_a_2_*'
        lf = scan_files(filename_prefix + file_id + '.parquet',
                        depth=2,
                        date_col=self.date_cols[file_id])
        cols = self._get_ignored_features(lf.columns, file_id)
        lf = lf.drop(cols)
        return lf

    def _cb_b_1(self, train: bool = True) -> pl.LazyFrame:
        """
        Read and process credit_bureau_b_1 data.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        filename_prefix = self.train_filename_prefix if train else self.test_filename_prefix
        file_id = 'credit_bureau_b_1'
        lf = scan_file(filename_prefix + file_id + '.parquet',
                       depth=1,
                       date_col=self.date_cols[file_id])
        cols = self._get_ignored_features(lf.columns, file_id)
        lf = lf.drop(cols)
        lf = self._aggregate_columns(lf, file_id)
        return lf

    def _person_1(self, train: bool = True) -> pl.LazyFrame:
        """
        Read and process person_1 data.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        filename_prefix = self.train_filename_prefix if train else self.test_filename_prefix
        file_id = 'person_1'
        lf = scan_file(filename_prefix + file_id + '.parquet',
                       depth=1,
                       date_col=self.date_cols[file_id])
        cols = self._get_ignored_features(lf.columns, file_id)
        lf = lf.drop(cols)
        lf = self._aggregate_columns(lf, file_id)
        # Education.
        edu_cols = [col for col in lf.columns if 'education' in col]
        lf = lf.with_columns(
            pl.col(edu_cols).map_elements(lambda x: self.education_grades[
                x] if x in self.education_grades else 0, return_dtype=pl.Int32).alias('education_2'))
        lf = lf.drop(edu_cols)
        return lf

    def _applprev_2(self, train: bool = True) -> pl.LazyFrame:
        """
        Read and process _applprev_2 data.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        filename_prefix = self.train_filename_prefix if train else self.test_filename_prefix
        table_id = 'applprev_2'
        lf = scan_file(filename_prefix + table_id + '.parquet',
                       depth=2,
                       date_col=self.date_cols[table_id])
        cols = self._get_ignored_features(lf.columns, table_id)
        lf = lf.drop(cols)
        return lf

    def _person_2(self, train: bool = True) -> pl.LazyFrame:
        """
        Read and process person_2 data.

        Parameters
        ----------
        train: bool
            Generate tables from train folder (True) or from test folder (False)

        Returns
        -------
        lf: pl.LazyFrame
        """
        filename_prefix = self.train_filename_prefix if train else self.test_filename_prefix
        file_id = 'person_2'
        lf = scan_file(filename_prefix + file_id + '.parquet',
                       depth=2,
                       date_col=self.date_cols[file_id])
        cols = self._get_ignored_features(lf.columns, file_id)
        lf = lf.drop(cols)
        return lf

    # endregion
