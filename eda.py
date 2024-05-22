from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from tqdm.notebook import tqdm


# region plots
def build_barplot_by_column(df: pd.DataFrame, column: str, title: str) -> None:
    """
    Строит нормированный barplot с долей людей в разрезе column по каждому сегменту.

    Parameters
    ----------
    df: pd.DataFrame
    column: str
        Название колонки по которой строить график
    title: str
        Подпись рисунка

    Returns
    -------
    None
    """
    in_segm_total = df.target.value_counts()  # нормировка на общее количество людей в сегментах
    grouped_by_segments = df.groupby(['target', column])['case_id'].count()
    #     нормировка на количество заполненных значений
    #     in_segm_total = grouped_by_segments.groupby(['target'])['case_id'].count()
    for s in df.target.unique():
        grouped_by_segments[s] /= in_segm_total[s] * 0.01
    grouped_by_segments = pd.DataFrame(grouped_by_segments)
    grouped_by_segments.rename(columns={"case_id": "Percentage"}, inplace=True)

    plt.figure(figsize=(15, 7))
    ax = sns.barplot(x=column,
                     y='Percentage',
                     data=grouped_by_segments,
                     hue='target', palette='rocket')

    for p in ax.patches:
        height = p.get_height()
        if height == 0:
            continue
        percentage = '{:.1f}%'.format(height)
        ax.annotate(percentage,
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points')

    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Percentage')
    plt.show()


def label(x: pd.Series, color: tuple, label: str) -> None:
    """
    Define a simple function to label the plot in axes coordinates

    Parameters
    ----------
    x: pd.Series

    color: tuple
        RGB

    label : str
        Category

    Returns
    -------
    None
    """
    ax = plt.gca()
    ax.text(.6,
            .2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes)


def draw_overlapping_densities(df: pd.DataFrame, target_col: str, category_col: str) -> None:
    """
    Plot Overlapping densities (‘ridge plot’)

    Parameters
    ----------
    df: pd.DataFrame

    target_col : str
        column name in the df

    Returns
    -------
    None
    """
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(len(df[category_col].unique()),
                                rot=np.random.random(),
                                light=.7)
    g = sns.FacetGrid(df,
                      row=category_col,
                      hue=category_col,
                      aspect=25,
                      height=.5,
                      palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot,
          target_col,
          bw_adjust=.5,
          clip_on=False,
          fill=True,
          alpha=1,
          linewidth=1.5)
    g.map(sns.kdeplot, target_col, clip_on=False, color="w", lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    g.map(label, target_col)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    plt.show()


# endregion

# region stat tests
def check_normal(x: pd.DataFrame, pvalue_threshold: float = 0.05) -> tuple[bool, float]:
    """
    Perform the Shapiro-Wilk test for normality.

    The Shapiro-Wilk test tests the null hypothesis that the
    data was drawn from a normal distribution.

    Parameters
    ----------
    x : array_like
        Array of sample data.

    pvalue_threshold : float
        default = 0.05

    Returns
    -------
    normal : bool
        True if x has a normal distribution (p-value >= pvalue_threshold)
    p-value : float
        The p-value for the hypothesis test.
    """
    pvalue = stats.shapiro(x).pvalue
    return pvalue >= pvalue_threshold, np.round(pvalue, 5)


def check_distributions_for_similarity(lf: pl.LazyFrame, col_name: str,
                                       **kwargs) -> None:
    """
    Print normality test results for target = 1 and 0
    If both are normal - perform Ttest and bootstrap test
    else perform Mann-Whitney U rank test on two independent samples

    Parameters
    ----------
    lf : pl.LazyFrame

    col_name : str
        feature name

    Returns
    -------
    None
    """
    df_1 = lf.filter(pl.col("target") == 1).select(col_name).collect()
    df_0 = lf.filter(pl.col("target") == 0).select(col_name).collect()
    check_normal_1 = check_normal(df_1)
    check_normal_0 = check_normal(df_0)
    for res, group_name in [(check_normal_1, '1'), (check_normal_0, '0')]:
        print(
            f"Распределение для признака {col_name} target = {group_name} {'' if res[0] else 'не '}нормальное, pvalue = {res[1]}"
        )
    df_0 = df_0.to_pandas().dropna()
    df_1 = df_1.to_pandas().dropna()
    if all([check_normal_0[0], check_normal_1[0]]):

        data_0 = df_0[col_name].values
        data_1 = df_1[col_name].values
        t_test(data_0, data_1)
    else:
        statistic, pvalue = mannwhitneyu(df_1, df_0)
        check_pvalue(pvalue)
        print(
            f" Mann–Whitney U Test for {col_name}: statistic={np.round(statistic, 5)}, p-value={np.round(pvalue, 5)}"
        )
    bootstrap_test(df_0, df_1, **kwargs)


def t_test(data1: np.ndarray, data2: np.ndarray) -> None:
    """
    Calculate the T-test for the means of *two independent* samples of scores.

    Parameters
    ----------
    data1: np.ndarray
    data2: np.ndarray

    Returns
    -------
    None
    """
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    statistic, pvalue = stats.ttest_ind(data1, data2)
    check_pvalue(pvalue)
    print(f'T test statistic = {statistic}, pvalue = {pvalue}')


def bootstrap_test(df0: pd.DataFrame, df1: pd.DataFrame, **kwargs) -> None:
    """
    Сравнение выборок при помощи Bootstrap.
    Parameters
    ----------
    df0: pd.DataFrame
    df1: pd.DataFrame

    Returns
    -------
    None
    """
    diff_sample, ci, pvalue = bootstrap_compare(df1, df0, **kwargs)
    fig, ax = plt.subplots(figsize=(15, 4))

    plt.title('Распределение разности группы 0 и 1')
    sns.kdeplot(diff_sample, fill=True)

    _, max_ylim = plt.ylim()
    plt.vlines(ci,
               color='orange',
               ymin=0,
               ymax=max_ylim,
               linestyle='--',
               linewidth=2)
    plt.vlines(0, ymin=0, ymax=max_ylim, linestyle='--', linewidth=2)
    print(f'CI = {ci}')


def check_pvalue(pvalue: float) -> None:
    """
    Вывод результата по значению p_value

    Parameters
    ----------
    pvalue: float

    Returns
    -------
    None

    """
    try:
        if pvalue >= 0.05:
            print(f'Средние схожи, p-value={pvalue}')
        else:
            print(f'Средние различны, p-value={pvalue}')
    except TypeError as ex:
        print(f'message: {ex}')


def bootstrap_compare(data1: pd.DataFrame,
                      data2: pd.DataFrame,
                      function: Callable = np.median,
                      alpha: float = 0.05,
                      count_generate: int = 9999,
                      random_state: int = 10) -> tuple:
    """
    Сравнение выборок при помощи Bootstrap

    Parameters
    ----------
    data1: pd.DataFrame
        Выборка 1
    data2: pd.DataFrame
        Выборка 2
    function: Callable
        Default is np.median
    alpha: float
        Уровень значимости
    count_generate: int
        Размер Bootstrap выборки
    random_state: int

    Returns
    -------
    out: tuple
        diff_sample, ci, p_value

    """
    np.random.seed(random_state)
    sample_size = max(len(data1), len(data2))

    diff_sample = []

    for i in tqdm(range(count_generate)):
        sample1 = data1.sample(sample_size, replace=True).values
        sample2 = data2.sample(sample_size, replace=True).values

        diff = function(sample1 - sample2)
        diff_sample.append(diff)

    # нижний и верхние квантили
    low = alpha / 2
    high = 1 - alpha / 2

    # интервал
    ci = (np.quantile(diff_sample, low), np.quantile(diff_sample, high))

    # p-value
    # Кумулятивная функция распределения (CDF) распределения вероятностей
    # содержит вероятности того, что случайная величина X меньше или равна X
    # 0 - в том случае, если разницы нет, среднее должно быть в нуле
    p1 = stats.norm.cdf(0, loc=np.mean(diff_sample), scale=np.std(diff_sample))
    p2 = 1 - p1
    p_value = min(p1, p2) * 2
    check_pvalue(p_value)

    return diff_sample, ci, p_value

# endregion
