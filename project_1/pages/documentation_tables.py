import dash_html_components as html
from utils import Header
import dash_core_components as dcc
import dash_html_components as html
from utils import Header, make_dash_table
import pandas as pd

def create_layout(app):
    return html.Div(
        [
            Header(app),
            # page 6
            html.Div(
                [

                    html.Div([
                    html.H5("Data analysis for variable 29 "),
                                html.Br([]),
                                html.P(
                                    "\
                                29: Which areas are you most eager to strengthen in order to reach your future ambition within 3 years:",
                                    style={"color": "#121214"},
                                    className="row",
                                ),
                    ]),

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
                                        id='corr-select29',
                                        options=[
                                            {'label': 'TOTAL ', 'value': 'TOT'},
                                            {'label': 'YES', 'value': 'YES'},
                                            {'label': 'NO', 'value': 'NO'},
                                            
                                        ],
                                        value='TOT'
                                    ),



                                    ]),
                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="Ex-Correlation29",
                                        
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
                                        id='varD129',
                                        options=[
                                            {'label': 'STRATEGY ', 'value': 'A'},
                                            {'label': 'ORGANIZATION', 'value': 'B'},
                                            {'label': 'INNOVATION PROJECT', 'value': 'C'},
                                            {'label': 'VALUE NETWORK', 'value': 'D'},
                                            {'label': 'RESULTS', 'value': 'E'},
                                        ],
                                        value='A'
                                    ),
                                    ]),
                                    html.Div([
                                        dcc.Dropdown(
                                        id='varD229',
                                        options=[
                                            {'label': 'STRATEGY ', 'value': 'A'},
                                            {'label': 'ORGANIZATION', 'value': 'B'},
                                            {'label': 'INNOVATION PROJECT', 'value': 'C'},
                                            {'label': 'VALUE NETWORK', 'value': 'D'},
                                            {'label': 'RESULTS', 'value': 'E'},
                                        ],
                                        value='B'
                                    ),
                                    ]),
                                    html.Div([
                                        dcc.Dropdown(
                                        id='varD329',
                                        options=[
                                            {'label': 'STRATEGY ', 'value': 'A'},
                                            {'label': 'ORGANIZATION', 'value': 'B'},
                                            {'label': 'INNOVATION PROJECT', 'value': 'C'},
                                            {'label': 'VALUE NETWORK', 'value': 'D'},
                                            {'label': 'RESULTS', 'value': 'E'},
                                        ],
                                        value='C'
                                    ),
                                    ]),
                                    html.Div([
                                        dcc.Dropdown(
                                        id='Ncluster129',
                                        options=[
                                            {'label': '2 ', 'value': '2'},
                                            {'label': '3', 'value': '3'},
                                            {'label': '4', 'value': '4'},
                                            {'label': '5', 'value': '5'},
                                            {'label': '6', 'value': '6'},
                                        ],
                                        value='3'
                                        ),
                                        

                                    ]),


                                    html.Div(
                                        [
                                            dcc.Graph(
                                        id="cluster229",
                                        
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
                                        id="metric1a29",
                                        
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
                                        id="metric2a29",
                                        
                                        config={"displayModeBar": False},
                                    ),
                                        ],
                                        
                                    ),
                                    
                                ],
                                className="six columns",
                            ),


                                    
                                    
                                ],
                                className="twelve columns",
                            )
                        ],
                        className="row ",
                    ),

                    # Row 1
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6("COMMENTS", className="subtitle padded"),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.P(
                                                "The exploration in this data set shows results based mainly on correlation, according to the hypotheses raised, it is verified that:"
                                            ),
                                            html.P(
                                                "-The relationship between A and E has a correlation of 0.59"
                                            ),
                                            html.P(
                                                "-B has the greatest impact on E according to the correlation matrix"
                                            ),
                                        ],
                                        style={"color": "#7a7a7a"},
                                    ),
                                ],
                                className="row",
                            ),
                            html.Div(
                                [
                                    html.H6("Resume", className="subtitle padded"),
                                    html.Br([]),
                                    html.Div(
                                        [
                                            html.Li("DATASET."),
                                            html.Li(
                                                "The most important areas are:"
                                            ),
                                            html.Li(
                                                "RESULTS, new solutions applications, benfits realization and effects of the innovation work"
                                            ),
                                        ],
                                        id="reviews-bullet-pts",
                                    ),
                                    
                                ],
                                className="row",
                            ),
                        ],
                        className="row ",
                    )
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
