import numpy as np
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import serial
from plotly.subplots import make_subplots
import time as t
import csv
import plotly.graph_objects as go
from Recurrence_Plot import get_recur_opt,get_opt_flow
from DeepInferenceModel import predict_normality

#initialising App
app = Dash(__name__)
subjectInfo = {}
subjectInfo['st'] = t.time()
subjectInfo['duration'] = 0
subjectData = {'Info':[],'Data':[]}
subjectData['flag'] = 0

# getting Information about Subject
getSubInfo = html.Div([
    html.Div('Name',style={'display':'inline-block','margin-right':20}),
    dcc.Input(id = 'subName',type='text'),
    html.Div('Height',style={'display':'inline-block','margin-right':20}),
    dcc.Input(id = 'subHeight',type = 'text'),
    html.Div('Weight',style={'display':'inline-block','margin-right':20}),
    dcc.Input(id = 'subWeight',type = 'text'),
    html.Div('Age',style={'display':'inline-block','margin-right':20}),
    dcc.Input(id = 'subAge',type = 'text'),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='afterSubmitMessage',children='Done')
],style={'textAlign': 'center'})
@app.callback(
    Output('afterSubmitMessage', 'children'),
    [Input('submit-val', 'n_clicks'),Input('subName', 'value'),
     Input('subHeight', 'value'),Input('subWeight', 'value'),
     Input('subAge', 'value')]
)
def update_output(nclicks, name, weight, height, age):
    subjectInfo['Name'] = name
    subjectInfo['Weight'] = weight
    subjectInfo['Height'] = height
    subjectInfo['Age'] = age
    if nclicks>0:
        outputString = f"Subject {name} with height {height} weight {weight} age {age}"
        return outputString

#getting live feed
getDuration = html.Div([
    html.Div('Duration (seconds)',style={'display':'inline-block','margin-right':20}),
    dcc.Input(id='feedDuration',type='number'),
    html.Button('Submit', id='buttonDuration', n_clicks=0),
    html.Div(id='divLiveFeed')
],style={'textAlign': 'center','margin-top':50})
@app.callback(
    Output('divLiveFeed','children'),
    [Input('feedDuration','value'),Input('buttonDuration','n_clicks')]
)
def update_output(duration,nclicks):
    if nclicks > 0:
        subjectInfo['duration'] = duration
        subjectData['Info'].append(subjectInfo)
        subjectInfo['st'] = t.time()
        return f'Duration {duration}s'

#setting up Live Feed
getLiveFeed = html.Div(
    html.Div([
        html.Div('Live Feed',style={'text-align':'left','font-size':30}),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1000,# in milliseconds
            n_intervals=20
        )
    ])
)
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n1):
    data = {
        'time': [],
        'ECG': []
    }
    ECGin, timeStamps = getData()
    data['time'] = timeStamps
    data['ECG'] = ECGin
    if (t.time() - subjectInfo['st']) <= subjectInfo['duration']:
        print('Recording...')
        subjectData['Data'].append(ECGin)
    else:
        if len(subjectData['Data']) != 0:
            print('Recording Done........')
            subjectData['flag'] = 1
            ECGData = sum(subjectData['Data'], [])
            name = subjectInfo['Name']
            weight = subjectInfo['Weight']
            height = subjectInfo['Height']
            age = subjectInfo['Age']
            with open(f'csv_{name}_{height}_{weight}_{age}', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(ECGData)

    fig = make_subplots(rows=1, cols=1)
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.append_trace({
        'x': data['time'],
        'y': data['ECG'],
        'name': 'Data',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)
    return fig

def getData(count=[1]):
    inputArr = []
    timeArr = []
    # trigger input
    bluetooth.write(b"1")
    while bluetooth.isOpen():
        blueInput = bluetooth.readline().rstrip().decode("utf-8")
        if blueInput == 'E':
            break
        else:
            inputArr.append(int(str(blueInput)))
    c = count[0]
    timeArr = np.linspace(c-1,c,len(inputArr))
    return inputArr, timeArr


#recording update
recordStatus = html.Div([
    dcc.Interval(id = 'statusInterval',max_intervals=-1,interval=1000),
    html.Div(id = 'statusDiv')
])
@app.callback(Output('statusDiv', 'children'),
              Input('statusInterval', 'n_intervals'))
def update_graph_live(n):
    if subjectData['flag'] == 1:
        return 'Recording Done .....'


#Recurrent Analysis
getRecurrentAnalysis = html.Div(children=[
    html.Button('Start Recurrent Analysis', id='startAnalysisButton', n_clicks=0),
    html.Div(id='ContainerAnalysis'),
])
@app.callback(
    Output('ContainerAnalysis', 'children'),
    Input('startAnalysisButton', 'n_clicks')
)
def update_output(n_clicks):
    if n_clicks > 0:
        print('Analysis started')
        AvgSubData = np.array(sum(subjectData['Data'], []))
        SubName = subjectInfo['Name']
        fig = px.line(x = np.linspace(0,1,AvgSubData.shape[0]),y = AvgSubData,labels={'x':'Time (ms)','y':'ECG Data'},
                      title = f'Time Series ECG data of Subject : {SubName}')
        fig.update_layout(
            autosize=False,
            width=600,
            height=300,
            title={
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        RecurData = get_recur_opt(sig = AvgSubData)
        figRecur = px.imshow(RecurData,title = 'Recurrence Plot')
        figRecur.update_layout(
            autosize=False,
            width=300,
            height=300,
            title={
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        epsilons,energies = get_opt_flow(AvgSubData)
        figEns = px.line(x = epsilons,y = energies, markers = True, title = 'Threshold Optimization',
                         labels = {'x':r'Threshold : Epsilon','y':r'Entropy'})
        figEns.update_layout(
            autosize=False,
            width=500,
            height=300,
            title={
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        figDiv = html.Div(children=[
            dcc.Graph(id='TimeSeries', style={'display': 'inline-block'}, figure=fig),
            dcc.Graph(id='RecurPlot', style={'display': 'inline-block'}, figure=figRecur),
            dcc.Graph(id='RecurPlot', style={'display': 'inline-block'}, figure=figEns)
        ])
        return figDiv

#Deep Inference
getDeepInference = html.Div(children=[
    html.Button('Start Machine Inference', id='startInferenceButton', n_clicks=0),
    html.Div(id='ContainerInference'),
])
@app.callback(
    Output('ContainerInference', 'children'),
    Input('startInferenceButton', 'n_clicks')
)
def update_output(n_clicks):
    if n_clicks > 0:
        print('Analysis started')
        AvgSubData = np.array(sum(subjectData['Data'], []))
        SubName = subjectInfo['Name']
        pred = predict_normality(AvgSubData)
        print(f'pred {pred}')
        fig = px.bar(x = ['Normal','Abnormal'],y = pred, title = 'Deep Inference')
        fig.update_layout(
            autosize=False,
            width=500,
            height=300,
            title={
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )
        figDiv = html.Div(children=[
            dcc.Graph(id='BarGraph', style={'display': 'inline-block'}, figure=fig)
        ])
        return figDiv

#App layout
app.layout = html.Div(children=[
    getSubInfo,
    getDuration,
    getLiveFeed,
    recordStatus,
    getRecurrentAnalysis,
    getDeepInference
])


if __name__ == '__main__':
    # setting up data acquisition
    print("Start")
    bluetooth = serial.Serial("COM8", 9600, timeout=1)
    print("Connected")
    bluetooth.flushInput()

    app.run_server(debug=False)