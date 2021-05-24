import dash_core_components as dcc
import dash_html_components as html
from utils import Header, make_dash_table
import pandas as pd
import pathlib
import plotly.graph_objs as go

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()

df_dividend = pd.read_csv(DATA_PATH.joinpath("df_dividend.csv"))
df_realized = pd.read_csv(DATA_PATH.joinpath("df_realized.csv"))
df_unrealized = pd.read_csv(DATA_PATH.joinpath("df_unrealized.csv"))

df_graph=pd.read_csv("dataX/CAD=X.csv")
df_graph2=pd.read_csv("dataX/JPY=X.csv")
#df_graph.value[0:1000]
def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 5
            html.Div(
                [


                   html.Div([
                    html.H5("Data analysis report to External variables"),
                                html.Br([]),
                                html.P(
                                    "\
                                This section shows results according for external variables.",
                                    style={"color": "#121214"},
                                    className="row",
                                ),
                    ]),
                    # KMeans
                    
                    html.Div(
                        [
                            
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "NEW RESULTS EXTERNAL VARIABLES "
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.Div([
                                        dcc.Dropdown(
                                        id='corr-select',
                                        options=[
                                            {'label': 'TOTAL ', 'value': 'TOTAL'},
                                            {'label': 'YES', 'value': 'YES'},
                                            {'label': 'NO', 'value': 'NO'},
                                            
                                        ],
                                        value='TOTAL'
                                    ),



                                    ]),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="Ex-Correlation",
                                        
                                        config={"displayModeBar": False},
                                    ),
                                        ],
                                        
                                    ),
                                    html.H6(
                                        [
                                            "Clustering external variables "
                                        ],
                                        className="subtitle padded",
                                    ),

                                  
                                    html.Div([
                                        dcc.Dropdown(
                                        id='Ncluster1',
                                        options=[
                                            {'label': '2 ', 'value': '2'},
                                            {'label': '3', 'value': '3'},
                                            {'label': '4', 'value': '4'},
                                            {'label': '5', 'value': '5'},
                                            {'label': '6', 'value': '6'},
                                        ],
                                        value='3'
                                        )

                                    ]),


                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="cluster2",
                                        
                                        config={"displayModeBar": False},
                                    ),
                                        ],
                                        
                                    ),

                                    html.Div(
                                [
                                    html.H6(
                                        [
                                            "Elbow rule"
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="metric1a",
                                        
                                        config={"displayModeBar": False},
                                    ),
                                        ],
                                        
                                    ),
                                    
                                ],
                                className="twelve columns",
                            ),
                                    html.Div(
                                [
                                    html.H6(
                                        [
                                            "Rule2 for the clusters"
                                        ],
                                        className="six columns",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="metric2a",
                                        
                                        config={"displayModeBar": False},
                                    ),
                                        ],
                                        
                                    ),
                                    
                                ],
                                className="six columns",
                            ),


                                    dcc.Dropdown(
                                        id='varaux1h',
                                        options=[
                                            {'label': 'A INNOVATION STRATEGY ', 'value': 'A'},
                                            {'label': 'B ORGANIZATION', 'value': 'B'},
                                            {'label': 'C INNOVATION PROJECT', 'value': 'C'},
                                            {'label': 'D VALUE NETWORK', 'value': 'D'},
                                            {'label': 'E RESULTS', 'value': 'E'},
                                        ],
                                        value='C'
                                    ),
                                    
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),

                  
                ],
                className="sub_page",
            ),
            
        ],
        className="page",
    )
