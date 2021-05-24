import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from factor_analyzer import FactorAnalyzer

from dash.dependencies import Input, Output

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
silhouette_scores = [] 
from pages import (
    overview,
    graph_analysis,
    correlation,
    indicators,
    documentation_tables,
)
xls = pd.ExcelFile('dataset_1.xlsx')
df1 = pd.read_excel(xls, 'part1')
df2 = pd.read_excel(xls, 'part2')
df3 = pd.read_excel(xls, 'yes')
df4 = pd.read_excel(xls, 'no')

df1 = df1.drop(labels=0, axis=0)

corr_1=df1.iloc[:, 3:8]
mxc1=corr_1.corr()
labels1=corr_1.columns
dfyes = df1[df1['Answer'] == 'YES']
dfno = df1[df1['Answer'] == 'NO']
corr_yes=dfyes.iloc[:, 3:8]
mxcyes=corr_yes.corr()
corr_no=dfno.iloc[:, 3:8]
mxcno=corr_no.corr()

df=df1.iloc[:,3:8]
kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
distortions = []
K1 = range(1,20)
for k in K1:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)

df=df1.iloc[:,3:8]
kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
silhouette_scores = [] 
K = range(2,10) 
X=df
for k in K:
    clusterer = KMeans(n_clusters=k)
    preds = clusterer.fit_predict(X)
    score = silhouette_score(X, preds)
    silhouette_scores.append(score)

### external variables###############3
xlsA = pd.ExcelFile('Database_v2.xlsx')
dfaA = pd.read_excel(xlsA, 'CHANGE1')

#kmeans2 = KMeans(n_clusters=3).fit(Exv)
#centroids2 = kmeans.cluster_centers_
####Part2#############
xlsa = pd.ExcelFile('Database_v2.xlsx')
df1aA = pd.read_excel(xlsa, 'CHANGE1')

df1k=dfaA.iloc[:,10:13]
kmeans = KMeans(n_clusters=3).fit(df1k)
centroids = kmeans.cluster_centers_
distortions1 = []
K1a = range(1,20)
for k in K1a:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df1k)
    distortions1.append(kmeanModel.inertia_)

df1k=dfaA.iloc[:,10:13]
kmeans = KMeans(n_clusters=3).fit(df1k)
centroids = kmeans.cluster_centers_
silhouette_scores1 = [] 
Ka = range(2,10) 
X=df1k
for k in Ka:
    clusterer = KMeans(n_clusters=k)
    preds = clusterer.fit_predict(X)
    score = silhouette_score(X, preds)
    silhouette_scores1.append(score)

###############question 29##############
corr_29=df2.iloc[:, 6:11]
mxc29=corr_29.corr()
labels29=corr_29.columns
dfyes29 = df2[df2['Answer'] == 'YES']
dfno29 = df2[df2['Answer'] == 'NO']
corr_yes29=dfyes29.iloc[:, 6:11]
mxcyes29=corr_yes29.corr()
corr_no29=dfno29.iloc[:, 6:11]
mxcno29=corr_no29.corr()
###########29 kmeans

df1k=df2.iloc[:, 6:11]
kmeans = KMeans(n_clusters=3).fit(df1k)
centroids = kmeans.cluster_centers_
distortions129 = []
K1a29 = range(1,20)
for k in K1a29:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df1k)
    distortions129.append(kmeanModel.inertia_)

df1k=df2.iloc[:, 6:11]
kmeans = KMeans(n_clusters=3).fit(df1k)
centroids = kmeans.cluster_centers_
silhouette_scores129 = [] 
Ka29 = range(2,10) 
X=df1k
for k in Ka29:
    clusterer = KMeans(n_clusters=k)
    preds = clusterer.fit_predict(X)
    score = silhouette_score(X, preds)
    silhouette_scores129.append(score)



app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server=app.server
app.config.suppress_callback_exceptions = True
app.layout=html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]

)



@app.callback(Output("page-content","children"),[Input("url","pathname")])
def display_page(pathname):
   
    if pathname == "/DATA_ANALYSIS/correlation":
        return correlation.create_layout(app)
    elif pathname == "/DATA_ANALYSIS/documentation_tables":
        return documentation_tables.create_layout(app)
    elif pathname == "/DATA_ANALYSIS/graph_analysis":
        return graph_analysis.create_layout(app)
    elif pathname == "/DATA_ANALYSIS/indicators":
        return indicators.create_layout(app)
    elif pathname == "/DATA_ANALYSIS/full-view":
        return(
            overview.create_layout(app),
            graph_analysis.create_layout(app),
            correlation.create_layout(app),
            indicators.create_layout(app),
            anomaly_detection.create_layout(app),
            documentation_tables.create_layout(app),
        )
    else:
        return overview.create_layout(app)


#############OVERVIEW#####################
@app.callback(
    Output("graph-1", "figure"), 
    [Input("buffer1", "value")])
def employes1(cols):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1.iloc[:, 1], y=df1.iloc[:, 2],
                    mode='markers',
                    name='markers'))
    #fig = px.scatter(df1, x=df1.iloc[:, 1], y=df1.iloc[:, 2])



    fig.update_traces(mode='lines+markers')
    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(xaxis={'visible': True, 'showticklabels': False})
    fig.update_layout(width=700, height=200, plot_bgcolor='rgb(255,255,255)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dddddd')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig['layout'].update(margin=dict(l=0,r=20,b=20,t=10))
    fig.update_traces(line=dict(color = "#0863ae"))
    fig.update_layout( 
    xaxis_title="Company",
    yaxis_title="NUmber of employees",
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    )
    )
    return fig


@app.callback(
    Output("graph-1-1", "figure"), 
    [Input("buffer1", "value")])
def plotfig(cols):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1.iloc[:, 2], y=df1.iloc[:, 8],
                    mode='markers',
                    name='markers'))

    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(xaxis={'visible': True, 'showticklabels': False})
    fig.update_layout(width=700, height=200, plot_bgcolor='rgb(255,255,255)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dddddd')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig['layout'].update(margin=dict(l=0,r=20,b=20,t=10))
    fig.update_traces(line=dict(color = "#0863ae"))
    fig.update_layout( 
    xaxis_title="Number of employees",
    yaxis_title="Answer",
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    )
    )
    return fig





@app.callback(
    Output("graph-2", "figure"), 
    [Input("buffer1", "value")])
def employes1(cols):
    df1 = pd.read_excel(xls, 'part1')  ##6-10
    df1=df1.iloc[:,3:8]

    df = px.data.tips()
    fig = go.Figure()
    #fig = px.box(df2, y=df2.iloc[:,0], points="all")
    fig.add_trace(go.Box( y=df1.iloc[:,0], boxpoints='all',name="A-INNOVATION-STRATEGY",marker_color='#2f323e',
        line_color='#2f323e'))
    fig.add_trace(go.Box( y=df1.iloc[:,1], boxpoints='all',name="B-ORGANIZATION",marker_color='rgb(7,40,89)',
        line_color='rgb(7,40,89)'))
    fig.add_trace(go.Box( y=df1.iloc[:,2], boxpoints='all',name="C-INNOVATION PROJECT",marker_color='rgb(9,56,125)',
        line_color='rgb(9,56,125)'))
    fig.add_trace(go.Box( y=df1.iloc[:,3], boxpoints='all',name="D-VALUE NETWORK",marker_color='rgb(8,81,156)',
        line_color='rgb(8,81,156)'))
    fig.add_trace(go.Box( y=df1.iloc[:,4], boxpoints='all',name="E-RESULTS",marker_color='rgb(107,174,214)',
        line_color='rgb(107,174,214)'))


    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(xaxis={'visible': True, 'showticklabels': False})
    fig.update_layout(width=700, height=200, plot_bgcolor='rgb(255,255,255)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dddddd')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig['layout'].update(margin=dict(l=0,r=20,b=20,t=10))
    fig.update_layout( 
    xaxis_title="Question type",
    yaxis_title="Response scale",
    legend_title="Question",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    )
    )
    return fig
@app.callback(
    Output("graph-3", "figure"), 
    [Input("buffer1", "value")])
def employes1(cols):
    df2 = pd.read_excel(xls, 'part2')  ##6-10
    df2=df2.iloc[:,6:11]

    df = px.data.tips()
    fig = go.Figure()
    #fig = px.box(df2, y=df2.iloc[:,0], points="all")
    fig.add_trace(go.Box( y=df2.iloc[:,0], boxpoints='all',name="STRATEGY",marker_color='#2f323e',))
    fig.add_trace(go.Box( y=df2.iloc[:,1], boxpoints='all',name="ORGANIZATION",marker_color='rgb(7,40,89)',))
    fig.add_trace(go.Box( y=df2.iloc[:,2], boxpoints='all',name="INNOVATION PROJECT",marker_color='rgb(9,56,125)',))
    fig.add_trace(go.Box( y=df2.iloc[:,3], boxpoints='all',name="VALUE NETWORK",marker_color='rgb(8,81,156)'))
    fig.add_trace(go.Box( y=df2.iloc[:,4], boxpoints='all',name="RESULTS",marker_color='rgb(107,174,214)'))
    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(xaxis={'visible': True, 'showticklabels': False})
    fig.update_layout(width=700, height=200, plot_bgcolor='rgb(255,255,255)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dddddd')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    fig['layout'].update(margin=dict(l=0,r=20,b=20,t=10))
    fig.update_layout( 
    xaxis_title="Question type",
    yaxis_title="Response scale",
    legend_title="Question",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    )
    )
    return fig
@app.callback(
    Output("graph-4A", "figure"), 
    [Input("buffer1", "value")])
def employes1(cols):
    df2bx = pd.read_excel(xls, 'part2')  ##6-10
    df2bx=df2bx.iloc[:,3:6]

    #df = px.data.tips()
    fig = go.Figure()
    #fig = px.box(df2, y=df2.iloc[:,0], points="all")
    fig.add_trace(go.Box( y=df2bx.iloc[:,0], boxpoints='all',name="was 3 years ago",marker_color='#2f323e',))
    fig.add_trace(go.Box( y=df2bx.iloc[:,1], boxpoints='all',name="is today",marker_color='rgb(7,40,89)',))
    fig.add_trace(go.Box( y=df2bx.iloc[:,2], boxpoints='all',name="desirable / planned position in 3 years",marker_color='rgb(9,56,125)',))

    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(xaxis={'visible': True, 'showticklabels': False})
    fig.update_layout(width=700, height=200, plot_bgcolor='rgb(255,255,255)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dddddd')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    fig['layout'].update(margin=dict(l=0,r=20,b=20,t=10))
    fig.update_layout( 
    xaxis_title="Question type",
    yaxis_title="Response scale",
    legend_title="Question",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    )
    )
    return fig
@app.callback(
    Output("graph-5", "figure"), 
    [Input("buffer1", "value")])
def employes1(cols):

    df2b5=dfaA.iloc[:,10:13]
    #df = px.data.tips()
    fig = go.Figure()
    #fig = px.box(df2, y=df2.iloc[:,0], points="all")
    fig.add_trace(go.Box( y=df2b5.iloc[:,0], boxpoints='all',name="NEW SOLUTION",marker_color='#2f323e',))
    fig.add_trace(go.Box( y=df2b5.iloc[:,1], boxpoints='all',name="IMPLEMENT",marker_color='rgb(7,40,89)',))
    fig.add_trace(go.Box( y=df2b5.iloc[:,2], boxpoints='all',name="BENEFITS REALIZATION",marker_color='rgb(9,56,125)',))

    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(xaxis={'visible': True, 'showticklabels': False})
    fig.update_layout(width=700, height=200, plot_bgcolor='rgb(255,255,255)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dddddd')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    fig['layout'].update(margin=dict(l=0,r=20,b=20,t=10))
    fig.update_layout( 
    xaxis_title="Question type",
    yaxis_title="Response scale",
    legend_title="Question",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    )
    )
    return fig
#############END OF OVERBVIEW


@app.callback(
    Output("correlation", "figure"), 
    [Input("correlation-selector", "value")])
def filter_heatmap(cols):
    matrix=[[1, 1, 1],
            [0.5, 0.10, 0],
            [0, 0.6, 0]]

    
    fig = go.Figure()
    if cols=='TOT':
        labels1=corr_1.columns
        fig = go.Figure(go.Heatmap(
            z= mxc1,
            x=labels1,
            y=labels1,
            colorscale= [
                [0, 'rgb(223,235,245)'],        #0
            
                [1., 'rgb(8,99,174)'],             #100000

            ],
            colorbar= dict(
                tick0= 0,
                tickmode= 'array',
                tickvals= [0, 1000, 10000, 100000]
            )
        ))
    if cols=='YES':
        labels1=corr_1.columns
        fig = go.Figure(go.Heatmap(
            z= mxcyes,
            x=labels1,
            y=labels1,
            colorscale= [
                [0, 'rgb(223,235,245)'],        #0
            
                [1., 'rgb(8,99,174)'],             #100000

            ],
            colorbar= dict(
                tick0= 0,
                tickmode= 'array',
                tickvals= [0, 1000, 10000, 100000]
            )
        ))
    if cols=='NO':
        labels1=corr_1.columns
        fig = go.Figure(go.Heatmap(
            z= mxcno,
            x=labels1,
            y=labels1,
            colorscale= [
                [0, 'rgb(223,235,245)'],        #0
            
                [1., 'rgb(8,99,174)'],             #100000

            ],
            colorbar= dict(
                tick0= 0,
                tickmode= 'array',
                tickvals= [0, 1000, 10000, 100000]
            )
        ))
    fig.update_layout(yaxis={'visible': False, 'showticklabels': False})
    fig.update_layout(xaxis={'visible': False, 'showticklabels': False})

    return fig

### CALBACK CORRELATION TWO VARIABLES

@app.callback(
    dash.dependencies.Output('graph-4', 'figure'),
    [dash.dependencies.Input('VARIABLE1', 'value'),
     dash.dependencies.Input('VARIABLE2', 'value'),
     ])
def update_x_timeseries(var1, var2):
    buff1=0
    buff2=0
    if var1=='A':
        buff1=0
    if var2=='A':
        buff2=0
    if var1=='B':
        buff1=1
    if var2=='B':
        buff2=1
    if var1=='C':
        buff1=2
    if var2=='C':
        buff2=2
    if var1=='D':
        buff1=3
    if var2=='D':
        buff2=3
    if var1=='E':
        buff1=4
    if var2=='E':
        buff2=4
    fig = px.scatter(corr_1, x=corr_1.iloc[:, buff1], y=corr_1.iloc[:, buff2])
    #fig.update_traces(mode='lines+markers')
    fig.update_layout(width=800, height=450, plot_bgcolor='rgb(255,255,255)')
    fig.update_xaxes(title_text=var1)
    fig.update_yaxes(title_text=var2)
    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(xaxis={'visible': True, 'showticklabels': False})
    fig.update_layout(width=700, height=200, plot_bgcolor='rgb(255,255,255)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dddddd')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig['layout'].update(margin=dict(l=0,r=20,b=20,t=10))
    fig.update_traces(line=dict(color = "#0863ae"))
    fig.update_layout( 

    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    )
    )
    return fig

'''

@app.callback(
    Output("graph-var", "figure"), 
    [Input("buffer", "value")])
def plotfig(cols):
    fig = px.scatter(df1, x=df1.iloc[:, 2], y=df1.iloc[:, 8])
    fig.update_traces(mode='markers')
    fig.update_layout(width=800, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig
'''

@app.callback(
    Output("graph-fa", "figure"), 
    [Input("buffer", "value")])
def plotfig(cols):
    c= df1.corr()
    xa=df1[df1.columns[2:7]] 
    fa = FactorAnalyzer()
    fa.fit(xa, 10)#Get Eigen values and plot them
    ev, v = fa.get_eigenvalues()
    ev
    #plt.plot(range(1,xa.shape[1]+1),ev)
    fig = px.scatter(x=range(1,xa.shape[1]+1), y=ev)
    fig.update_traces(mode='lines+markers')
 
    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(xaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(width=700, height=200, plot_bgcolor='rgb(255,255,255)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dddddd')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig['layout'].update(margin=dict(l=0,r=20,b=20,t=10))
    fig.update_traces(line=dict(color = "#0863ae"))
    fig.update_layout( 
    xaxis_title="X",
    yaxis_title="Y",
    legend_title="Factor Analysis",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="black"
    )
    )
    return fig

@app.callback(
    Output("taska", "figure"), 
    [Input("buffer", "value")])
def plotfig(cols):

    xa=df1[df1.columns[2:7]] 

##########SCATTER#############3
    df=df1
    y=df1['A INNOVATION STRATEGY - 4: As a whole / in summary, the plans and governing documents we have for innovation provide the right conditions for innovation work']
    x=df1.iloc[:, 4:7]
    dfy=df1['A INNOVATION STRATEGY - 4: As a whole / in summary, the plans and governing documents we have for innovation provide the right conditions for innovation work']
    dfy=dfy.to_frame()
    y=dfy
    N = 102
    ya=df1['A INNOVATION STRATEGY - 4: As a whole / in summary, the plans and governing documents we have for innovation provide the right conditions for innovation work']
    x=df1.iloc[:, 3:7]
    
    x0=x.iloc[:,0]
    x1=x.iloc[:,1]
    x2=x.iloc[:,2]
    x3=x.iloc[:,3]
    varx=np.linspace(0, 102, N)
    fig = go.Figure()
    varx=df1['#']
    # Add traces
    fig.add_trace(go.Scatter(x=varx, y=x0,
                        mode='markers',
                        name='A'))
    fig.add_trace(go.Scatter(x=varx, y=x1,
                        mode='markers',
                        name='B'))
    fig.add_trace(go.Scatter(x=varx, y=x2,
                        mode='markers',
                        name='C'))
    fig.add_trace(go.Scatter(x=varx, y=x3,
                        mode='markers',
                        name='D'))
    #fig.update_layout(xaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    fig.update_layout(showlegend=True)
    fig.update_xaxes(title_text='# of authority')
    #plt.plot(range(1,xa.shape[1]+1),ev)
    '''
    fig = px.scatter(x=range(1,xa.shape[1]+1), y=ev)
    fig.update_traces(mode='markers')
    fig.update_layout(width=800, height=450, plot_bgcolor='rgb(255,255,255)')
    '''
    
    return fig

@app.callback(
    Output("task1", "figure"), 
    [Input("task3", "value")])
def plotfig(cols):
    dfy=df1['A INNOVATION STRATEGY - 4: As a whole / in summary, the plans and governing documents we have for innovation provide the right conditions for innovation work']
    dfy=dfy.to_frame()
    x=df1.iloc[:, 4:7]
    y=dfy
    # Split the data into training/testing sets
    X_train = x.iloc[1:100,:]
    X_test = x.iloc[50:100,:]
    Y_train = y.iloc[1:100,:]
    Y_test = y.iloc[50:100,:]
    # Split the targets into training/testing sets

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)
    ####part2
    Y_test=Y_test.to_numpy()
    Y_test=Y_test.reshape(50,1)
    y_pred = Y_pred.reshape(50,1)
    print(Y_test.shape)
    print(Y_pred.shape)
    #df = pd.DataFrame({'Y_p':y_pred})
    df = pd.DataFrame(np.hstack((Y_test, y_pred)))

    x_range = np.linspace(X_test.min(), X_test.max(), 100)
    y_range = regr.predict(x_range)

    yp=y_range.reshape(100,1)
    xp=x_range[:,2].reshape(100,1)

    df = pd.DataFrame(np.hstack((xp, yp)))
    y_range
    print(xp.shape)
    print(yp.shape)


    ###############model knn#################

    knn_dist = KNeighborsRegressor(10, weights='distance')
    knn_dist.fit(X_train, Y_train)


    y_dist = knn_dist.predict(x_range)


    yp2=y_dist.reshape(100,1)
    xp2=x_range[:,2].reshape(100,1)

    df2 = pd.DataFrame(np.hstack((xp2, yp2)))



    fig = px.scatter(x=df.iloc[:,0], y=df.iloc[:,1], labels={'x': 'X_test', 'y': 'Y_prediction'})
    #fig = px.scatter(x=X_test.iloc[:,0], y=df.iloc[:,1], labels={'x': 'X_test', 'y': 'Y_prediction'})
    fig.add_traces(go.Scatter(x=df.iloc[:,0], y=df.iloc[:,1],name='Regression-line'))

    fig.add_traces(go.Scatter(x=df2.iloc[:,0], y=df2.iloc[:,1], name='KNN-MODEL'))

    fig.add_traces(go.Scatter(x=X_train.iloc[:,0], y=df.iloc[:,1], mode='markers',name='B'))
    fig.add_traces(go.Scatter(x=X_train.iloc[:,1], y=df.iloc[:,1], mode='markers',name='C'))
    fig.add_traces(go.Scatter(x=X_train.iloc[:,2], y=df.iloc[:,1], mode='markers',name='D'))


    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig
'''
@app.callback(
    Output("task2", "figure"), 
    [Input("buffer", "value")])
def plotfig(cols):

    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig
'''
@app.callback(
    Output("task2", "figure"), 
    [Input("task3", "value")])

def plotfig(cols):
    dfy=df1['B ORGANIZATION - 12: As a whole / in summary, the organization provides the right conditions for innovation work']
    dfy=dfy.to_frame()
    x=df1.iloc[:, 5:7]
    y=dfy
    # Split the data into training/testing sets
    X_train = x.iloc[1:100,:]
    X_test = x.iloc[50:100,:]
    Y_train = y.iloc[1:100,:]
    Y_test = y.iloc[50:100,:]
    # Split the targets into training/testing sets

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)
    ####part2
    Y_test=Y_test.to_numpy()
    Y_test=Y_test.reshape(50,1)
    y_pred = Y_pred.reshape(50,1)
    print(Y_test.shape)
    print(Y_pred.shape)
    #df = pd.DataFrame({'Y_p':y_pred})
    df = pd.DataFrame(np.hstack((Y_test, y_pred)))

    x_range = np.linspace(X_test.min(), X_test.max(), 100)
    y_range = regr.predict(x_range)

    yp=y_range.reshape(100,1)
    xp=x_range[:,1].reshape(100,1)

    df = pd.DataFrame(np.hstack((xp, yp)))
    y_range
    print(xp.shape)
    print(yp.shape)


    ###############model knn#################

    knn_dist = KNeighborsRegressor(10, weights='distance')
    knn_dist.fit(X_train, Y_train)


    y_dist = knn_dist.predict(x_range)


    yp2=y_dist.reshape(100,1)
    xp2=x_range[:,1].reshape(100,1)

    df2 = pd.DataFrame(np.hstack((xp2, yp2)))



    fig = px.scatter(x=df.iloc[:,0], y=df.iloc[:,1], labels={'x': 'X_test', 'y': 'Y_prediction'})
    #fig = px.scatter(x=X_test.iloc[:,0], y=df.iloc[:,1], labels={'x': 'X_test', 'y': 'Y_prediction'})
    fig.add_traces(go.Scatter(x=df.iloc[:,0], y=df.iloc[:,1],name='Regression-line'))

    fig.add_traces(go.Scatter(x=df2.iloc[:,0], y=df2.iloc[:,1], name='KNN-MODEL'))

    fig.add_traces(go.Scatter(x=X_train.iloc[:,0], y=df.iloc[:,1], mode='markers',name='C-INNOVATION PROJECT'))
    fig.add_traces(go.Scatter(x=X_train.iloc[:,1], y=df.iloc[:,1], mode='markers',name='D-VALUE NETWORK'))
    #fig.add_traces(go.Scatter(x=X_train.iloc[:,2], y=df.iloc[:,1], mode='markers',name='D'))


    #fig.show()
    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig
@app.callback(
    Output("task3", "figure"), 
    [Input("task5", "value")])

def plotfig(cols):

    dfy=df1['E RESULTS - 27: As a whole / in summary, our innovation work creates the results that are sought\n']
    dfy=dfy.to_frame()
    x=df1.iloc[:, 5:8]
    y=dfy
    # Split the data into training/testing sets
    X_train = x.iloc[1:100,:]
    X_test = x.iloc[50:100,:]
    Y_train = y.iloc[1:100,:]
    Y_test = y.iloc[50:100,:]
    # Split the targets into training/testing sets

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)
    ####part2
    Y_test=Y_test.to_numpy()
    Y_test=Y_test.reshape(50,1)
    y_pred = Y_pred.reshape(50,1)
    print(Y_test.shape)
    print(Y_pred.shape)
    #df = pd.DataFrame({'Y_p':y_pred})
    df = pd.DataFrame(np.hstack((Y_test, y_pred)))

    x_range = np.linspace(X_test.min(), X_test.max(), 100)
    y_range = regr.predict(x_range)

    yp=y_range.reshape(100,1)
    xp=x_range[:,2].reshape(100,1)

    df = pd.DataFrame(np.hstack((xp, yp)))
    y_range
    print(xp.shape)
    print(yp.shape)


    ###############model knn#################

    knn_dist = KNeighborsRegressor(5, weights='distance')
    knn_dist.fit(X_train, Y_train)


    y_dist = knn_dist.predict(x_range)


    yp2=y_dist.reshape(100,1)
    xp2=x_range[:,2].reshape(100,1)

    df2 = pd.DataFrame(np.hstack((xp2, yp2)))



    fig = px.scatter(x=df.iloc[:,0], y=df.iloc[:,1], labels={'x': 'X_test', 'y': 'Y_prediction'})
    #fig = px.scatter(x=X_test.iloc[:,0], y=df.iloc[:,1], labels={'x': 'X_test', 'y': 'Y_prediction'})
    fig.add_traces(go.Scatter(x=df.iloc[:,0], y=df.iloc[:,1],name='Regression-line'))

    fig.add_traces(go.Scatter(x=df2.iloc[:,0], y=df2.iloc[:,1], name='KNN-MODEL'))

    fig.add_traces(go.Scatter(x=X_train.iloc[:,0], y=df.iloc[:,1], mode='markers',name='B-ORGANIZATION'))
    fig.add_traces(go.Scatter(x=X_train.iloc[:,1], y=df.iloc[:,1], mode='markers',name='C-INNOVATION PROJECT'))
    fig.add_traces(go.Scatter(x=X_train.iloc[:,2], y=df.iloc[:,1], mode='markers',name='D-VALUE-NETWORK'))

    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig

@app.callback(
    Output("task5", "figure"), 
    [Input("task6", "value")])

def plotfig(cols):
    dfy=df1['E RESULTS - 27: As a whole / in summary, our innovation work creates the results that are sought\n']
    dfy=dfy.to_frame()
    x=df1['A INNOVATION STRATEGY - 4: As a whole / in summary, the plans and governing documents we have for innovation provide the right conditions for innovation work']
    x=x.to_frame()

    y=dfy
    # Split the data into training/testing sets
    X_train = x.iloc[1:100]
    X_test = x.iloc[50:100]
    Y_train = y.iloc[1:100]
    Y_test = y.iloc[50:100]
    # Split the targets into training/testing sets

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)
    ####part2
    Y_test=Y_test.to_numpy()
    Y_test=Y_test.reshape(50,1)
    y_pred = Y_pred.reshape(50,1)
    print(Y_test.shape)
    print(Y_pred.shape)
    #df = pd.DataFrame({'Y_p':y_pred})
    df = pd.DataFrame(np.hstack((Y_test, y_pred)))

    x_range = np.linspace(X_test.min(), X_test.max(), 100)
    y_range = regr.predict(x_range)

    yp=y_range.reshape(100,1)
    xp=x_range[:,0].reshape(100,1)

    df = pd.DataFrame(np.hstack((xp, yp)))
    y_range
    print(xp.shape)
    print(yp.shape)


    ###############model knn#################

    knn_dist = KNeighborsRegressor(80, weights='uniform')
    knn_dist.fit(X_train, Y_train)


    y_dist = knn_dist.predict(x_range)


    yp2=y_dist.reshape(100,1)
    xp2=x_range[:,0].reshape(100,1)

    df2 = pd.DataFrame(np.hstack((xp2, yp2)))



    fig = px.scatter(x=df.iloc[:,0], y=df.iloc[:,1], labels={'x': 'X_test', 'y': 'Y_prediction'})
    #fig = px.scatter(x=X_test.iloc[:,0], y=df.iloc[:,1], labels={'x': 'X_test', 'y': 'Y_prediction'})
    fig.add_traces(go.Scatter(x=df.iloc[:,0], y=df.iloc[:,1],name='Regression-line'))

    fig.add_traces(go.Scatter(x=df2.iloc[:,0], y=df2.iloc[:,1], name='KNN-MODEL'))

    fig.add_traces(go.Scatter(x=X_train.iloc[:,0], y=df.iloc[:,1], mode='markers',name='E-RESULTS'))

    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig

@app.callback(
    dash.dependencies.Output('k-means', 'figure'),
    [dash.dependencies.Input('varD1', 'value'),
     dash.dependencies.Input('varD2', 'value'),
     dash.dependencies.Input('varD3', 'value'),
     dash.dependencies.Input('Ncluster', 'value'),
     ])
def update_x_timeseries(var1, var2,var3, cluster):

    df=df1.iloc[:,3:8]
    kmeans = KMeans(n_clusters=int(cluster)).fit(df)
    centroids = kmeans.cluster_centers_
    if var1=='A':
        vx1=df.iloc[:,0]
    if var2=='A':
        vx2=df.iloc[:,0]
    if var3=='A':
        vx3=df.iloc[:,0]    
    if var1=='B':
        vx1=df.iloc[:,1]
    if var2=='B':
        vx2=df.iloc[:,1]
    if var3=='B':
        vx3=df.iloc[:,1]
    if var1=='C':
        vx1=df.iloc[:,2]
    if var2=='C':
        vx2=df.iloc[:,2]
    if var3=='C':
        vx3=df.iloc[:,2]    
    if var1=='D':
        vx1=df.iloc[:,3]
    if var2=='D':
        vx2=df.iloc[:,3]
    if var3=='D':
        vx3=df.iloc[:,3]
    if var1=='E':
        vx1=df.iloc[:,4]
    if var2=='E':
        vx2=df.iloc[:,4]
    if var3=='E':
        vx3=df.iloc[:,4]         
    #fig = px.scatter_3d(df, x=vx1, y=vx2, z=vx3,color=kmeans.labels_.astype(float))
    
    fig = go.Figure(data=[go.Scatter3d(
        x=vx1,
        y=vx2,
        z=vx3,
        mode='markers',
        marker=dict(
            size=5,
            color=kmeans.labels_.astype(float),                # set color to an array/list of desired values
            colorscale='Viridis',
            opacity=0.8
    )
    )])


    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')

    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(xaxis={'visible': True, 'showticklabels': True})

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dddddd')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig['layout'].update(margin=dict(l=0,r=20,b=20,t=10))
    return fig


@app.callback(
    Output("metric1", "figure"), 
    [Input("task6", "value")])

def plotfig(cols):
    
    fig = px.scatter(x=K1, y=distortions, labels={'x': 'Distortion', 'y': 'Optimal K'})
    fig.update_traces(mode='lines+markers')
    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig
@app.callback(
    Output("metric2", "figure"), 
    [Input("task6", "value")])
def plotfig(cols):
    
    fig = px.scatter(x=K, y=silhouette_scores, labels={'x': 'K - Number of Clusters', 'y': 'Silhouette score'})
    fig.update_traces(mode='lines+markers')
    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig
###############################33EXTERNAL######################33

@app.callback(
    Output("Ex-Correlation", "figure"), 
    [Input('corr-select', 'value')])

def plotfig(cols):
    dfauxa=df1aA.iloc[:,7:10]
    mxc1a=dfauxa.corr()
    dfyes = df1aA[df1aA['ANSWER'] == 'YES']
    dfno = df1aA[df1aA['ANSWER'] == 'NO']

    corr_yes=dfyes.iloc[:,7:10]
    mxcyes1=corr_yes.corr()

    corr_no=dfno.iloc[:,7:10]
    mxcno1=corr_no.corr()

    fig = go.Figure()
    if cols=='TOTAL':
        labels1x=dfauxa.columns
        fig = go.Figure(go.Heatmap(
            z= mxc1a,
            x=labels1x,
            y=labels1x,
            colorscale= [
                [0, 'rgb(223,235,245)'],        #0
            
                [1., 'rgb(8,99,174)'],             #100000

            ],
            colorbar= dict(
                tick0= 0,
                tickmode= 'array',
                tickvals= [0, 1000, 10000, 100000]
            )
        ))
    if cols=='YES':
        labels1=dfauxa.columns
        fig = go.Figure(go.Heatmap(
            z= mxcyes1,
            x=labels1,
            y=labels1,
            colorscale= [
                [0, 'rgb(223,235,245)'],        #0
            
                [1., 'rgb(8,99,174)'],             #100000

            ],
            colorbar= dict(
                tick0= 0,
                tickmode= 'array',
                tickvals= [0, 1000, 10000, 100000]
            )
        ))
    if cols=='NO':
        labels1=dfauxa.columns
        fig = go.Figure(go.Heatmap(
            z= mxcno1,
            x=labels1,
            y=labels1,
            colorscale= [
                [0, 'rgb(223,235,245)'],        #0
            
                [1., 'rgb(8,99,174)'],             #100000

            ],
            colorbar= dict(
                tick0= 0,
                tickmode= 'array',
                tickvals= [0, 1000, 10000, 100000]
            )
        ))
    fig.update_layout(yaxis={'visible': False, 'showticklabels': False})
    fig.update_layout(xaxis={'visible': False, 'showticklabels': False})


    return fig



@app.callback(
    Output("cluster2", "figure"), 
    [Input('Ncluster1', 'value')])

def plotfig(VALUE):
    Exv=dfaA.iloc[:,10:13]
    
    #df=Exv
    kmeans2 = KMeans(n_clusters=int(VALUE)).fit(Exv)
    centroids2 = kmeans2.cluster_centers_

    fig = px.scatter_3d(Exv, x=Exv.iloc[:,0], y=Exv.iloc[:,1], z=Exv.iloc[:,2],color=kmeans2.labels_.astype(float))
    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig

@app.callback(
    Output("metric1a", "figure"), 
    [Input("Ncluster1", "value")])

def plotfig(cols):
    
    fig = px.scatter(x=K1a, y=distortions1, labels={'x': 'Distortion', 'y': 'Optimal K'})
    fig.update_traces(mode='lines+markers')
    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig
@app.callback(
    Output("metric2a", "figure"), 
    [Input("Ncluster1", "value")])
def plotfig(cols):
    
    fig = px.scatter(x=Ka, y=silhouette_scores1, labels={'x': 'K - Number of Clusters', 'y': 'Silhouette score'})
    fig.update_traces(mode='lines+markers')
    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig

##################VARIABLE 29#############################


@app.callback(
    Output("Ex-Correlation29", "figure"), 
    [Input("corr-select29", "value")])
def filter_heatmap(cols):
    matrix=[[1, 1, 1],
            [0.5, 0.10, 0],
            [0, 0.6, 0]]

    
    fig = go.Figure()
    if cols=='TOT':
        labels1=corr_29.columns
        fig = go.Figure(go.Heatmap(
            z= mxc29,
            x=labels1,
            y=labels1,
            colorscale= [
                [0, 'rgb(223,235,245)'],        #0
            
                [1., 'rgb(8,99,174)'],             #100000

            ],
            colorbar= dict(
                tick0= 0,
                tickmode= 'array',
                tickvals= [0, 1000, 10000, 100000]
            )
        ))
    if cols=='YES':
        labels1=corr_29.columns
        fig = go.Figure(go.Heatmap(
            z= mxcyes29,
            x=labels1,
            y=labels1,
            colorscale= [
                [0, 'rgb(223,235,245)'],        #0
            
                [1., 'rgb(8,99,174)'],             #100000

            ],
            colorbar= dict(
                tick0= 0,
                tickmode= 'array',
                tickvals= [0, 1000, 10000, 100000]
            )
        ))
    if cols=='NO':
        labels1=corr_29.columns
        fig = go.Figure(go.Heatmap(
            z= mxcno29,
            x=labels1,
            y=labels1,
            colorscale= [
                [0, 'rgb(223,235,245)'],        #0
            
                [1., 'rgb(8,99,174)'],             #100000

            ],
            colorbar= dict(
                tick0= 0,
                tickmode= 'array',
                tickvals= [0, 1000, 10000, 100000]
            )
        ))
    fig.update_layout(yaxis={'visible': False, 'showticklabels': False})
    fig.update_layout(xaxis={'visible': False, 'showticklabels': False})

    return fig

@app.callback(
    dash.dependencies.Output('cluster229', 'figure'),
    [dash.dependencies.Input('varD129', 'value'),
     dash.dependencies.Input('varD229', 'value'),
     dash.dependencies.Input('varD329', 'value'),
     dash.dependencies.Input('Ncluster129', 'value'),
     ])
def update_x_timeseries(var1, var2,var3, cluster):

    df=df2.iloc[:, 6:11]
    kmeans = KMeans(n_clusters=int(cluster)).fit(df)
    centroids = kmeans.cluster_centers_
    if var1=='A':
        vx1=df.iloc[:,0]
    if var2=='A':
        vx2=df.iloc[:,0]
    if var3=='A':
        vx3=df.iloc[:,0]    
    if var1=='B':
        vx1=df.iloc[:,1]
    if var2=='B':
        vx2=df.iloc[:,1]
    if var3=='B':
        vx3=df.iloc[:,1]
    if var1=='C':
        vx1=df.iloc[:,2]
    if var2=='C':
        vx2=df.iloc[:,2]
    if var3=='C':
        vx3=df.iloc[:,2]    
    if var1=='D':
        vx1=df.iloc[:,3]
    if var2=='D':
        vx2=df.iloc[:,3]
    if var3=='D':
        vx3=df.iloc[:,3]
    if var1=='E':
        vx1=df.iloc[:,4]
    if var2=='E':
        vx2=df.iloc[:,4]
    if var3=='E':
        vx3=df.iloc[:,4]         
    #fig = px.scatter_3d(df, x=vx1, y=vx2, z=vx3,color=kmeans.labels_.astype(float))
    
    fig = go.Figure(data=[go.Scatter3d(
        x=vx1,
        y=vx2,
        z=vx3,
        mode='markers',
        marker=dict(
            size=5,
            color=kmeans.labels_.astype(float),                # set color to an array/list of desired values
            colorscale='Viridis',
            opacity=0.8
    )
    )])


    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')

    fig.update_layout(yaxis={'visible': True, 'showticklabels': True})
    fig.update_layout(xaxis={'visible': True, 'showticklabels': True})

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#dddddd')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig['layout'].update(margin=dict(l=0,r=20,b=20,t=10))
    return fig


@app.callback(
    Output("metric1a29", "figure"), 
    [Input("Ncluster129", "value")])

def plotfig(cols):
    
    fig = px.scatter(x=K1a29, y=distortions129, labels={'x': 'Distortion', 'y': 'Optimal K'})
    fig.update_traces(mode='lines+markers')
    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig
@app.callback(
    Output("metric2a29", "figure"), 
    [Input("Ncluster129", "value")])
def plotfig(cols):
    
    fig = px.scatter(x=Ka29, y=silhouette_scores129, labels={'x': 'K - Number of Clusters', 'y': 'Silhouette score'})
    fig.update_traces(mode='lines+markers')
    fig.update_layout(width=700, height=450, plot_bgcolor='rgb(255,255,255)')
    return fig


###############END VARIABLE 29##############################

if __name__ == "__main__":
    app.run_server(debug=True)









