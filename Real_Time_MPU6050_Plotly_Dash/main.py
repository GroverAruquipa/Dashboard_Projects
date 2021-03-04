import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.express as px
import random
import plotly.graph_objs as go
from collections import deque
import dash_bootstrap_components as dbc
import base64
X = deque(maxlen=20)
X.append(1)
Y = deque(maxlen=20)
Y.append(1)
X2=deque(maxlen=10)
Y2=deque(maxlen=10)
X.append(1)
Y.append(1)

app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], meta_tags=[{"name": "viewport", "content": "width=device-width"}])
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)
app.layout = html.Div(
    [
    html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url('index.png'),
                            id="plotly-image",
                            style={
                                "height": "80px",
                                "width": "auto",
                                "margin-bottom": "px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Accelerometer and Gyro supervisor",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "real-time", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Learn More", id="learn-more-button"),
                            href="https://plot.ly/dash/pricing/",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
    html.Div([
        

        
        html.Div([
            dcc.Graph(id='live-graph', animate=True),
            dcc.Interval(
            id='graph-update',
            interval=1*1000
        )
        ] ,className="pretty_container"), 
        
       
        html.Div([
            html.H3("Graphics controls, parameter selection for the accelerometer",
            style={"margin-bottom": "0px"},),
            html.H4("Selector of variable",
            style={"margin-bottom": "0px"},),
            dcc.Dropdown(id='accels', className='as',
                options=[
                    {'label': 'X-Acceleration', 'value': 'Xa'},
                    {'label': 'y-Acceleration', 'value': 'Ya'},
                    {'label': 'Z-Acceleration', 'value': 'Za'}

                ],
                value=['Xa','Ya'],
                #placeholder=''
            ),
            html.H4("Selector of Color",
            style={"margin-bottom": "0px"},),
            dcc.Dropdown(id='accelscolor', className='as',
                options=[
                    {'label': 'Color1', 'value': 'Xa'},
                    {'label': 'Color2', 'value': 'Ya'},
                    {'label': 'Color3', 'value': 'Za'}

                ],
                value=['Xa','Ya'],
                #placeholder=''
            ),
            html.H4("Selector of type of graphic",
            style={"margin-bottom": "0px"},),
            dcc.Dropdown(id='accelstype', className='as',
                options=[
                    {'label': 'scatter1', 'value': 'Xa'},
                    {'label': 'full', 'value': 'Ya'},
                    {'label': 'normal', 'value': 'Za'}

                ],
                value=['Xa','Ya'],
                #placeholder=''
            ),

        ], style = { 'width': 'auto'},
        className="pretty_container",
        
        ),

    ], style={'columnCount': 2},)
    ,
    html.Div([
        html.Div([
            dcc.Graph(id='live-graph2', 
                    animate=True,
                     ),
            dcc.Interval(
            id='graph-update2',
            interval=1*1000
        ),
        ]),
        html.Div([
            html.H3("Graphics controls, parameter selection for the gyroscope",
            style={"margin-bottom": "0px"},),
            html.H4("Selector of variable",
            style={"margin-bottom": "0px"},),
             dcc.Dropdown(id='Gyro', className='as',
                options=[
                    {'label': 'X-Gyro', 'value': 'Xg'},
                    {'label': 'y-Gyro', 'value': 'Yg'},
                    {'label': 'Z-Gyro', 'value': 'Zg'}

                ],
                value=['Xa','Ya'],
                #placeholder=''
            ),
              html.H4("Selector of Color",
            style={"margin-bottom": "0px"},),
            dcc.Dropdown(id='accelscolor2', className='as',
                options=[
                    {'label': 'Color1', 'value': 'Xa'},
                    {'label': 'Color2', 'value': 'Ya'},
                    {'label': 'Color3', 'value': 'Za'}

                ],
                value=['Xa','Ya'],
                #placeholder=''
            ),
            html.H4("Selector of type of graphic",
            style={"margin-bottom": "0px"},),
            dcc.Dropdown(id='accelstype2', className='as',
                options=[
                    {'label': 'scatter1', 'value': 'Xa'},
                    {'label': 'full', 'value': 'Ya'},
                    {'label': 'normal', 'value': 'Za'}

                ],
                value=['Xa','Ya'],
                #placeholder=''
            ),
        ])

    ],style={'columnCount': 2},className="pretty_container",),
    

    ],
    
)

@app.callback(Output('live-graph', 'figure'),
              Input('graph-update', 'n_intervals'),
              Input('accels', 'value')
              
              )
def update_graph_scatter(input_data,accels):
    
    if accels== 'Xa':
        X.append(X[-1]+1)
        Y.append(Y[-1]+Y[-1]*random.uniform(-0.1,0.1))
    elif accels=='Ya':
        X.append(X[-1]+1)
        Y.append(Y[-1]+Y[-1]*random.uniform(-5,5))
    else: 
        X.append(X[-1]+1)
        Y.append(Y[-1]+Y[-1]*random.uniform(-1,1))
    data = plotly.graph_objs.Scatter(
                x=list(X),
                y=list(Y),
                name='Scatter',
                mode= 'lines+markers'
                )

    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                yaxis=dict(range=[min(Y),max(Y)]),)}

@app.callback(Output('live-graph2', 'figure'),
              Input('graph-update2', 'n_intervals'),
              Input('Gyro','value'))
def update_graph_scatter(input_data,Gyro):

    if Gyro== 'Xg':
        X.append(X[-1]+1)
        Y.append(Y[-1]+Y[-1]*random.uniform(-2,2))
    elif Gyro=='Yg':
        X.append(X[-1]+1)
        Y.append(Y[-1]+Y[-1]*random.uniform(-5,5))
    else: 
        X.append(X[-1]+1)
        Y.append(Y[-1]+Y[-1]*random.uniform(-1,1))
    
    data = plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode= 'markers',
            fill='tozeroy',
            fillcolor='rgb(127, 166, 238)'
            )
    
    return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                yaxis=dict(range=[min(Y),max(Y)]),)}




if __name__ == '__main__':
    app.run_server(debug=True)