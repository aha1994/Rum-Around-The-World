import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import plotly.express as px


def heatMap(country='jamaica', number_notes=25, attr_='Taste_Notes'):
    # load tasting df
    rum_tasting = pd.read_csv('Data/rum.csv')
    
    # subset df based on attr_
    rum_tasting = rum_tasting[~pd.isna(rum_tasting[attr_])]
    rum_tasting[attr_] = [eval(l) for l in rum_tasting[attr_]]
    
    # get top notes for specific country
    country_tasting = rum_tasting[rum_tasting['Country'] == country]
    
    # get the top notes for chosen country
    tastes = defaultdict(int)
    num_tastes = number_notes

    for l in country_tasting[attr_]:
        for t in l:
            tastes[t] += 1

    tastes = dict(tastes)
    tastes = {k:v for k,v in sorted(tastes.items(), key = lambda x:x[1], reverse=True)}
    topX = list(tastes.keys())[0:num_tastes]
   
    u = (pd.get_dummies(pd.DataFrame(list(rum_tasting[attr_])), prefix='', prefix_sep='')
           .groupby(level=0, axis=1)
           .sum())
    v = u.T.dot(u)
    v.values[(np.r_[:len(v)], ) * 2] = 0
    
    v_topX = v.loc[v.index.isin(topX), v.columns.isin(topX)]
    
    
    
    u_X = (pd.get_dummies(pd.DataFrame(list(country_tasting[attr_])), prefix='', prefix_sep='')
       .groupby(level=0, axis=1)
       .sum())
    v_X = u_X.T.dot(u_X)
    v_X.values[(np.r_[:len(v_X)], ) * 2] = 0
    X_topX = v_X.loc[v_X.index.isin(topX), v_X.columns.isin(topX)]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # plot the specific country data
    sns.heatmap(X_topX, ax=axes[0], cmap = 'rocket_r')
    axes[0].set_title(f'Top {number_notes} {attr_}: {country.title()}')
            
    # plot all rum data
    sns.heatmap(v_topX, ax=axes[1], cmap = 'rocket_r')
    axes[1].set_title(f'Top {number_notes} {attr_}: All Rums')

    return fig


def formatFlavorProfile(profile, flavors):
    return pd.DataFrame([[int(f in profile) for f in flavors]], columns = flavors)

def makeScatter(df, country):
    fig = px.scatter(df, x="Price", y='Rating')
    fig.update_layout(title_text=f'Price vs Rating for {country} Rums', title_x=0.4)
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)

    return fig

def makeHist(df, metric):
    fig = px.histogram(df, x=metric, height=565, width = 800)
    if metric == 'Rating':
        fig.update_traces(xbins=dict( # bins used for histogram
                start=0.0,
                end=10.0,
                size=0.5
            ))
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
        fig.update_layout(title={'text': '<b>Distribution of Rum Ratings</b>'}, title_x=0.4)

    return fig

def makeBar(df):
    num_countries = 10
    data = df[df['Country'].isin(list(df['Country'].value_counts()[0:num_countries].index))]['Country'].value_counts()
    df_countries = pd.DataFrame({
    'Country': data.index,
    'Count': data.values
    })

    fig = px.bar(df_countries, x='Country', y='Count', height=600, width=800)
    fig.update_traces(marker_color='rgb(255,175,0)', marker_line_color='rgb(255,20,0)',
                    marker_line_width=1.5, opacity=0.6)
    fig.update_layout(title={'text': '<b>Top 10 Rum Producing Countries</b>'}, title_x=0.4)
    fig.update_xaxes(tickangle=-90)

    return fig