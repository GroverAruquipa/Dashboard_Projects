import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from utils import Header, make_dash_table

import pandas as pd
import pathlib

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../data").resolve()


df_fund_facts = pd.read_csv(DATA_PATH.joinpath("df_fund_facts.csv"))
df_price_perf = pd.read_csv(DATA_PATH.joinpath("df_price_perf.csv"))


def create_layout(app):
    # Page layouts
    return html.Div(
        [
            html.Div([Header(app)]),
            # page 1
            html.Div(
                [
                    # Row 3
                    html.Div(
                        [
                            html.Div( 
                                [
                                    html.H5("Data analysis report"),
                                    html.Br([]),
                                    html.P(
                                        "\
                                    In this section is presented some important characteristics that describe the data",
                                        style={"color": "#ffffff"},
                                        className="row",
                                    ),
                                ],
                                className="product",id='buffer1',
                            )
                        ],
                        className="row",
                    ),
                    # Row 4
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        ["Important-data"], className="subtitle padded"
                                    ),
                                    html.Table(make_dash_table(df_fund_facts)),
                                ],
                                className="seven columns",
                            ),
                            
                        ],
                        className="row",
                        style={"margin-bottom": "35px"},
                    ),
                    html.Div(
                                [
                                    html.H6(
                                        "Graph of number of employees per company",
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-1",
                                        
                                        
                                    ),
                                ],
                                
                            ),
                    html.Div(
                                [
                                    html.H6(
                                        "Number of employees versus Response",
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-1-1",
                                        
                                        
                                    ),
                                ],
                                
                            ),
                    # Row 5
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H6(
                                        "BOX PLOT responses PART 1: INNOVATION STRATEGY 4, ORGANIZATION 12, INNOVATION PROJECT 16, VALUE NETWORK 16 and RESULTS-27",
                                        className="subtitle padded",
                                    ),
                                    dcc.Graph(
                                        id="graph-2",
                                        
                                    ),
                                ],
                               # className="six columns",
                            ),
                         
                            html.Div(
                                [
                                    html.H6(
                                        "BOX PLOT PART2:29: Which areas are you most eager to strengthen in order to reach your future ambition within 3 years:", className="subtitle padded"
                                    ),
                                    dcc.Graph(
                                        id="graph-3",
                                      
                                    ),
                                ],
                                #className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "BOX PLOT PART3:28: Where do you see your authority:", className="subtitle padded"
                                    ),
                                    dcc.Graph(
                                        id="graph-4A",
                                      
                                    ),
                                ],
                                #className="six columns",
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "BOX PLOT NEW-RESULT-EXTERNAL: 25: In our innovation work with a focus on external results (eg new products, services and forms of collaboration), we as a whole succeed in meeting new challenges / opportunities by:", className="subtitle padded"
                                    ),
                                    dcc.Graph(
                                        id="graph-5",
                                      
                                    ),
                                ],
                                #className="six columns",
                            ),
                        ],
                       # className="row ",
                    ),
                ],
                className="sub_page",
            ),
        ],
        className="page",
    )
