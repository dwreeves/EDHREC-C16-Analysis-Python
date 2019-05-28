import numpy as np
import pandas as pd
from scipy import stats
from patsy import dmatrices
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import logit
from functools import lru_cache

COMMANDERS = ['atraxa', 'breya', 'kynaios', 'saskia', 'yidris']

DEP_VARS = [
    'has_mana_confluence',
    'has_city_of_brass',
    'has_either',
    'has_both'
]

@lru_cache()
def import_decks():
    # Import all data
    df = pd.DataFrame({'commander' : []})
    for cmdr in COMMANDERS:
        df = df.append(
            pd.read_csv('data/%s.csv' % cmdr, sep='|', header=None)
        )
        df['commander'] = df['commander'].fillna(cmdr)

    # Clean and compress data
    df = df.rename(columns={0 : 'deck', 1 : 'card'})
    prefix_len = len('http://tappedout.net/mtg-decks/')
    df['deck'] = df['deck'].str[prefix_len:]
    df = df.set_index('commander', 'deck', 'card')
    
    # Return City of Brass / Mana Conf boolean
    df['has_mana_confluence'] = df.groupby('deck')['card'] \
        .transform(lambda x: max(x == 'Mana Confluence'))
    df['has_city_of_brass'] = df.groupby('deck')['card'] \
        .transform(lambda x: max(x == 'City of Brass'))
    df['has_either'] = df['has_mana_confluence'] | df['has_city_of_brass']
    df['has_both'] = df['has_mana_confluence'] & df['has_city_of_brass']

    return df

@lru_cache()
def import_prices():
    df = pd.read_excel('data/mtgallprices.xlsx', sheet_name='Sheet1')
    rename_columns = {
        'Card Name' : 'card',
        'Fair Trade Price' : 'price'
    }
    df = df.rename(columns=rename_columns)
    df.head()
    df = df[['card', 'price']]
    df = df.set_index('card')

    # Filter out basic lands and unpriced cards
    basic_lands = ['Swamp', 'Forest', 'Mountain', 'Island', 'Plains']
    df = df[~df['card'].isin(basic_lands)]
    df = df[df['price'] != 0]

    # Return lowest price
    df = df.groupby('card').min()

    return df

@lru_cache()
def import_precons():
    df = pd.DataFrame({'commander' : []})
    for cmdr in COMMANDERS:
        df = df.append(
            pd.read_stata('data/precon_%s.dta' % cmdr)
        )
        df['commander'] = df['commander'].fillna(cmdr)
    df = df.set_index('commander', 'card')
    return df

def analysis_precon_overlap():
    # Import data and measure precon differences
    mrg = import_precons()
    mrg['count'] = 1
    df = import_decks().merge(mrg, on=['card', 'commander'], how='left')

    # Measure overlap and differences
    series_overlap = \
        df.groupby(['deck', 'commander'])['count'].sum() \
        .rename('precon_overlap')
    series_diff = \
        100 - df.groupby(['deck', 'commander'])['count'].sum()
    series_diff = series_diff.rename('precon_diff')

    # Create collapsed DataFrame
    df_collapsed = pd.concat([series_diff, series_overlap], axis=1)
    for v in DEP_VARS:
        series = df.groupby(['deck', 'commander'])[v].max().rename(v)
        df_collapsed = pd.concat([df_collapsed, series], axis=1)

    # Summary statistics 
    print(series_diff.groupby('commander').describe())
    
    # Univariate probit models (ungrouped)
    model = {}
    depvar_series = sm.add_constant(df_collapsed['precon_overlap'])
    for v in DEP_VARS:
        model[v] = \
            sm.Probit(
                df_collapsed[v],
                depvar_series
            )
        res = model[v].fit()
        print(res.summary())

    # Univariate probit models (grouped by commander)
    for cmdr in COMMANDERS:
        df_grouped = df_collapsed.query('commander == "%s"' % cmdr)
        depvar_series = sm.add_constant(df_grouped['precon_overlap'])
        for v in DEP_VARS:
            model_grped = \
                sm.Probit(
                    df_grouped[v],
                    depvar_series
                )
            res = model_grped.fit()
            print(res.summary())

    # Summary table
    tbl_actual = df_collapsed \
        .reset_index() \
        .groupby('precon_overlap') \
        .mean()

    tbl_density = series_overlap \
        .reset_index() \
        .groupby('precon_overlap') \
        .count() \
        [['deck']].rename(columns={'deck' : 'count'})
    tbl_density['density'] = tbl_density['count'] / len(df_collapsed)

    tbl = pd.DataFrame(index=range(90))
    tbl.index.name = 'precon_overlap'
    tbl = tbl.merge(tbl_actual, on='precon_overlap', how='left')
    tbl = tbl.merge(tbl_density, on='precon_overlap', how='left')
    tbl = tbl.reset_index()
        
    tbl['const'] = 1

    x_mat = \
        pd.concat([
            tbl['const'],
            tbl['precon_overlap']
        ], axis=1)

    b1 = model['has_either'].fit().predict(x_mat).rename('has_either_est')
    b2 = model['has_both'].fit().predict(x_mat).rename('has_both_est')

    tbl = pd.concat([tbl, b1, b2], axis=1)
    
    return tbl

def analysis_precon_overlap_graph_1(tbl, filepath):
    
    # Probability of Having City of Brass and/or Mana Confluence
    # as a Function of Overlap with Precons

    # Initialize graph
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = fig.add_axes(ax1.get_position(), frameon=False)
    lines = list()

    # Add data
    tbl['has_either_est'].plot(
        style='-b',
        ax=ax2
    )
    tbl['has_both_est'].plot(
        style='-r',
        ax=ax2
    )
    tbl['has_either'].plot(
        style='.b',
        ax=ax2
    )
    tbl['has_both'].plot(
        style='.r',
        ax=ax2
    )
    tbl['density'].plot(
        ax=ax1,
        color='0.87'
    )
    ax1.fill_between(
        tbl['precon_overlap'],
        0, tbl['density'],
        color='0.87',
        label='Sample density'
    )

    # Add options
    ax1.patch.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.xaxis.set_visible(False)

    ax2.set_yticklabels(['{:,.0%}'.format(x) for x in ax2.get_yticks()])
    ax2.set_xlabel('Number of cards that overlap with precon')

    plt.text(
        -10, -0.18,
        'Source: EDHREC.com, TappedOut.net, and author\'s calculations',
        ha='left',
        fontsize=7
    )
    plt.suptitle(
        'Probability of Having City of Brass and/or Mana Confluence',
        fontsize=14
    )
    plt.title(
        'as a Function of Overlap with Precons',
        fontsize=10
    )
    ax2.legend(
        ax2.get_lines(),
        ['Has either', 'Has both'],
        loc='upper right',
        bbox_to_anchor=(-0.0525, 0, 1, 1),
        frameon=False
    )
    ax1.legend(
        ax1.get_lines(),
        ['Sample density'],
        loc='upper right',
        bbox_to_anchor=(0, -0.09, 1, 1),
        frameon=False
    )

    plt.savefig(filepath)

if __name__ == '__main__':
    import logging
    import sys
    sys.stdout = open('output/log.txt', 'w')
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    tbl_precon_overlap = analysis_precon_overlap()
    
    analysis_precon_overlap_graph_1(
        tbl=tbl_precon_overlap,
        filepath='output/analysis_precon_overlap_5c_pains.png'
    )
    