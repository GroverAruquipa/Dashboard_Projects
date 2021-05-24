import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from utils import Header, make_dash_table
import pandas as pd
import pathlib

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()


df_current_prices = pd.read_csv(DATA_PATH.joinpath("df_current_prices.csv"))
df_hist_prices = pd.read_csv(DATA_PATH.joinpath("df_hist_prices.csv"))
df_avg_returns = pd.read_csv(DATA_PATH.joinpath("df_avg_returns.csv"))
df_after_tax = pd.read_csv(DATA_PATH.joinpath("df_after_tax.csv"))
df_recent_returns = pd.read_csv(DATA_PATH.joinpath("df_recent_returns.csv"))

df_dividend = pd.read_csv(DATA_PATH.joinpath("df_dividend.csv"))
df_realized = pd.read_csv(DATA_PATH.joinpath("df_realized.csv"))
df_unrealized = pd.read_csv(DATA_PATH.joinpath("df_unrealized.csv"))

df_graph=pd.read_csv("dataX/CAD=X.csv")
df_graph2=pd.read_csv("dataX/JPY=X.csv")
#df_graph = pd.read_csv(DATA_PATH.joinpath("df_graph.csv")) ##CHANGE
#df_graph=pd.read_csv("/home/grover/Documents/pythondoc/dashboards/FOREX1-DASH/dataX/CAD=X.csv")
#df_graph2=pd.read_csv("/home/grover/Documents/pythondoc/dashboards/FOREX1-DASH/dataX/JPY=X.csv")

def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 2
            html.Div(
                [
                html.Div([
                    html.H5("Data analysis report"),
                                html.Br([]),
                                html.P(
                                    "\
                                This section shows the results and tools for the variables INNOVATION STRATEGY, ORGANIZATION, INNOVATION PROJECT, VALUE NETWORK and RESULTS.",
                                    style={"color": "#121214"},
                                    className="row",
                                ),
                    ]),
                    # Row
                  html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Correlation graph"], className="subtitle padded"
                                    ),
                                    html.P(
                                        [
                                            "In this section is presented the correlation between different variables"
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 1.1 ## Matrix correlation
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Br([]),
                                    html.H6(
                                        ["Correlation matrix"],
                                        className="subtitle tiny-header padded",
                                    ),
                                    html.Div([
                                        dcc.Dropdown(
                                        id='correlation-selector',
                                        options=[
                                            {'label': 'Total Correlation', 'value': 'TOT'},
                                            {'label': 'Answer YES', 'value': 'YES'},
                                            {'label': 'Answer NO', 'value': 'NO'}
                                        ],
                                        value='TOT'
                                    ),
                                    ]),
                                    html.Div(
                                        [
                                            dcc.Graph(id="correlation"
                                                        ),
                                        ],
                                        style={"overflow-x": "auto"},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Resume"],
                                        className="subtitle tiny-header padded",
                                    )
                                ],
                                className=" twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 4
                    html.Div(
                        [
                            
                            html.Div(
                                [html.Table(make_dash_table(df_realized))],
                                className="six columns",
                            ),
                        ],
                        className="row ",
                    ),
                    # New Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Br([]),
                                    html.H6(
                                        ["Correlation between two variables"],
                                        className="subtitle tiny-header padded",
                                    ),
                                    html.Div([
                                        dcc.Dropdown(
                                        id='VARIABLE1',
                                        options=[
                                            {'label': 'A INNOVATION STRATEGY ', 'value': 'A'},
                                            {'label': 'B ORGANIZATION', 'value': 'B'},
                                            {'label': 'C INNOVATION PROJECT', 'value': 'C'},
                                            {'label': 'D VALUE NETWORK', 'value': 'D'},
                                            {'label': 'E RESULTS', 'value': 'E'},
                                        ],
                                        value='A'
                                    ),
                                    dcc.Dropdown(
                                        id='VARIABLE2',
                                        options=[
                                            {'label': 'A INNOVATION STRATEGY ', 'value': 'A'},
                                            {'label': 'B ORGANIZATION', 'value': 'B'},
                                            {'label': 'C INNOVATION PROJECT', 'value': 'C'},
                                            {'label': 'D VALUE NETWORK', 'value': 'D'},
                                            {'label': 'E RESULTS', 'value': 'E'},
                                        ],
                                        value='A'
                                    ),
                                    ]
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="graph-4",     
                                    ),
                                        ],
                                        
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),


                   
                      html.Div([
                                html.H6("Factor analysis", className="subtitle padded"),
                                html.Div([]),
                                dcc.Graph(
                                        id="graph-fa",
                                     
                                        config={"displayModeBar": False},
                                    ),
                            
                            ]),
                    html.Div(id='buffer'),

                    # Row 3
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "Eigenvalues of Factor analysis"
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.Div(
                                        [
                                            html.Table(
                                                make_dash_table(df_avg_returns),
                                                className="tiny-header",
                                            )
                                        ],
                                        style={"overflow-x": "auto"},
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),

                    html.Div(
                        [
                            
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "CLUSTERING ANALYSIS "
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.H6(
                                        [
                                            "Variable selectors"
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.Div([
                                        dcc.Dropdown(
                                        id='varD1',
                                        options=[
                                            {'label': 'A INNOVATION STRATEGY ', 'value': 'A'},
                                            {'label': 'B ORGANIZATION', 'value': 'B'},
                                            {'label': 'C INNOVATION PROJECT', 'value': 'C'},
                                            {'label': 'D VALUE NETWORK', 'value': 'D'},
                                            {'label': 'E RESULTS', 'value': 'E'},
                                        ],
                                        value='A'
                                    ),
                                    ]),
                                    html.Div([
                                        dcc.Dropdown(
                                        id='varD2',
                                        options=[
                                            {'label': 'A INNOVATION STRATEGY ', 'value': 'A'},
                                            {'label': 'B ORGANIZATION', 'value': 'B'},
                                            {'label': 'C INNOVATION PROJECT', 'value': 'C'},
                                            {'label': 'D VALUE NETWORK', 'value': 'D'},
                                            {'label': 'E RESULTS', 'value': 'E'},
                                        ],
                                        value='B'
                                    ),
                                    ]),
                                    html.Div([
                                        dcc.Dropdown(
                                        id='varD3',
                                        options=[
                                            {'label': 'A INNOVATION STRATEGY ', 'value': 'A'},
                                            {'label': 'B ORGANIZATION', 'value': 'B'},
                                            {'label': 'C INNOVATION PROJECT', 'value': 'C'},
                                            {'label': 'D VALUE NETWORK', 'value': 'D'},
                                            {'label': 'E RESULTS', 'value': 'E'},
                                        ],
                                        value='C'
                                    ),
                                    ]),
                                    html.Div([
                                        dcc.Dropdown(
                                        id='Ncluster',
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
                                        id="k-means",
                                        
                                        config={"displayModeBar": True},
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
                                        id="metric1",
                                        
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
                                        id="metric2",
                                        
                                        config={"displayModeBar": True},
                                    ),
                                        ],
                                        
                                    ),
                                    
                                ],
                                className="six columns",
                            ),
                                    
                                ],
                                className="six columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 4 Taska plot regressions
                    html.Div(
                        [
                            
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "Graph of the points"
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="taska",
                                        
                                        config={"displayModeBar": False},
                                    ),
                                        ],
                                        
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 5 Task 1 plot regressions
                    html.Div(
                        [
                            
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "A affects B, C and D"
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="task1",
                                        
                                        config={"displayModeBar": False},
                                    ),
                                        ],
                                        
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 6 Task 2 plot regressions
                    html.Div(
                        [
                            
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "B affects C and D "
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="task2",
                                        
                                        config={"displayModeBar": False},
                                    ),
                                        ],
                                        
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 7 Task 3 plot regressions
                    html.Div(
                        [
                            
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "B, C and D as a package affect E "
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="task3",
                                        
                                        config={"displayModeBar": False},
                                    ),
                                        ],
                                        
                                    ),
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),
                    # Row 8 Task 5 plot regressions
                    html.Div(
                        [
                            
                            html.Div(
                                [
                                    html.H6(
                                        [
                                            "Relationship between A and E "
                                        ],
                                        className="subtitle padded",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="task5",
                                        
                                        config={"displayModeBar": False},
                                    ),
                                        ],
                                        
                                    ),
                                    html.Div(dcc.Graph(id="task6")),
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



