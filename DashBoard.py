# Import required libraries
import numpy as np
import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from dash import no_update


# Create a dash application
app = dash.Dash(__name__)

# Clear the layout and do not display exception till callback gets executed
app.config.suppress_callback_exceptions = True

# Read the data into pandas dataframe
df = pd.read_csv(f'tweets2.csv')

def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}

"""Compute graph data for creating sentiment analysis report 

Function that takes data as input and create 4 dataframes based on the grouping condition to be used for plottling charts and grphs.

Argument:
     
    df: Filtered dataframe
    
Returns:
   Dataframes to create graph. 
"""
def compute_data_choice_1(df):
    # Language Count
    Language_count = df['Language'].value_counts(normalize=True).reset_index()
    df_draw = Language_count.rename(columns={"index": "Language", "Language": "Percent"})
    df_draw.loc[df_draw['Percent'] < 0.02, 'Language'] = 'other'
    pie_data = df_draw.groupby('Language')['Percent'].sum().reset_index()
    # Source Count
    Source_count = df['Source'].value_counts().reset_index()
    df_draw2 = Source_count.rename(columns={"index": "Source", "Source": "count"})
    df_draw2.loc[df_draw2['count'] < 100, 'Source'] = 'other'
    bar_data1 = df_draw2.groupby('Source')['count'].sum().reset_index()
    # Sentiment Count
    Sentiment_count = df['Sentiment'].value_counts().reset_index()
    df_draw3 = Sentiment_count.rename(columns={"index": "Sentiment", "Sentiment": "count"})
    bar_data2 = df_draw3
    # Correlation
    heat_data = df.select_dtypes(include=np.number).corr()
    
    return pie_data, bar_data1, bar_data2, heat_data


"""Compute graph data for creating sentiment analysis report

This function takes in data as an input and performs computation for creating charts and plots.

Arguments:
    df: Input data.
    
Returns:
    Series to create graph
"""
def compute_data_choice_2(df):
    hist_data1 = df['Negative']
    hist_data2 = df['Neutral']
    hist_data3 = df['Positive']
    hist_data4 = df['Length of tweet']
    return hist_data1, hist_data2, hist_data3, hist_data4


# Application layout
app.layout = html.Div(children=[
                                html.H1('Sentiment Analysis of COVID-19 Tweets', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 24}),
                                # Create an outer division 
                                html.Div(
                                    # Add an division
                                    html.Div([
                                        # Create an division for adding dropdown helper text for report type
                                        html.Div(
                                            [
                                            html.H2('Report Type:', style={'margin-right': '2em'}),
                                            ]
                                        ),
                                        # Add a dropdown
                                        dcc.Dropdown(id='input-type', options=[{'label': 'Data distribution (Language, Source, Sentiment)', 'value': 'OPT1'},
                                        {'label': 'Sentiment distribution and length of tweets', 'value': 'OPT2'}], placeholder='Select a report type',
                                        style={'width':'80%', 'padding':'3px', 'font-size':'20px', 'text-align-last':'center'})
                                    # Place them next to each other using the division style
                                    ], style={'display':'flex'})  
                                          ),
                                
                                # Add Computed graphs
    
                                html.Div([
                                        html.Div([ ], id='plot1'),
                                        html.Div([ ], id='plot2')
                                ], style={'display': 'flex'}),
                                
                                # Add a division with two empty divisions inside
                                html.Div([
                                        html.Div([ ], id='plot3'),
                                        html.Div([ ], id='plot4')
                                ], style={'display': 'flex'})
                                ])

# Callback function definition
# Add 4 ouput components
@app.callback( [Output(component_id='plot1', component_property='children'),
                Output(component_id='plot2', component_property='children'),
                Output(component_id='plot3', component_property='children'),
                Output(component_id='plot4', component_property='children')],
               [Input(component_id='input-type', component_property='value')
                ],
               # Holding output state till user enters all the form information. In this case, it will be chart type
               [State("plot1", 'children'), State("plot2", "children"),
                State("plot3", "children"), State("plot4", "children")
               ])
# Add computation to callback function and return graph
def get_graph(chart, children1, children2, c3, c4):
       
        if chart == 'OPT1':
            # Compute required information for creating graph from the data
            pie_data, bar_data1, bar_data2, heat_data = compute_data_choice_1(df)
            
            # Languages by percent
            pie_fig = px.pie(pie_data, values='Percent', names='Language', title="Languages by percent")
            # Sources by total number
            bar_fig1 = px.bar(bar_data1, x='count', y='Source', title='Sources by total number')
            
            # Sentiment by total number
            bar_fig2 = px.bar(bar_data2, x='count', y='Sentiment', title='Sentiment by total number')
            
            # Correlation between variables
            heat_fig = go.Figure(data=go.Heatmap(df_to_plotly(heat_data))).update_layout(title="Correlation between variables")           
            
            # Return dcc.Graph component to the empty division
            return [dcc.Graph(figure=pie_fig),
                    dcc.Graph(figure=bar_fig1),
                    dcc.Graph(figure=bar_fig2),
                    dcc.Graph(figure=heat_fig)
                   ]
        else:
            # Compute required information for creating graph from the data
            hist_data1, hist_data2, hist_data3, hist_data4 = compute_data_choice_2(df)
            
            # Create graph
            hist_fig1 = px.histogram(hist_data1, nbins=100, title='Histogram of negative tweets')
            hist_fig2 = px.histogram(hist_data2, nbins=100, title='Histogram of neutral tweets')
            hist_fig3 = px.histogram(hist_data3, nbins=100, title='Histogram of positive tweets')
            hist_fig4 = px.histogram(hist_data4, nbins=15, title='Histogram of tweet length')
            
            return[dcc.Graph(figure=hist_fig1), 
                   dcc.Graph(figure=hist_fig2), 
                   dcc.Graph(figure=hist_fig3), 
                   dcc.Graph(figure=hist_fig4)]


# Run the app
if __name__ == '__main__':
    app.run_server()