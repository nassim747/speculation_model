# app.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from investment_model import InvestmentModel

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server  # For deployment on platforms like Heroku

# App layout
app.layout = html.Div([
    html.H1("Speculative Investment Bubble Simulator", style={'textAlign': 'center'}),
    
    # Parameter controls
    html.Div([
        html.Div([
            html.H3("Population A Parameters"),
            html.Label("Initial Stock X Investors (%)"),
            dcc.Slider(id="init-a-slider", min=0, max=1, step=0.05, value=0.1, marks={0: '0%', 0.5: '50%', 1: '100%'}),
            
            html.Label("Risk Aversion"),
            dcc.Slider(id="risk-a-slider", min=1, max=3, step=0.1, value=2, marks={1: '1', 2: '2', 3: '3'}),
            
            html.Label("Social Influence Factor"),
            dcc.Slider(id="social-a-slider", min=0, max=1, step=0.05, value=0.6, marks={0: '0', 0.5: '0.5', 1: '1'})
        ], className="six columns"),
        
        html.Div([
            html.H3("Population B Parameters"),
            html.Label("Initial Stock X Investors (%)"),
            dcc.Slider(id="init-b-slider", min=0, max=1, step=0.05, value=0.1, marks={0: '0%', 0.5: '50%', 1: '100%'}),
            
            html.Label("Risk Aversion"),
            dcc.Slider(id="risk-b-slider", min=1, max=3, step=0.1, value=2, marks={1: '1', 2: '2', 3: '3'}),
            
            html.Label("Social Influence Factor"),
            dcc.Slider(id="social-b-slider", min=0, max=1, step=0.05, value=0.6, marks={0: '0', 0.5: '0.5', 1: '1'})
        ], className="six columns")
    ], className="row"),
    
    html.Div([
        html.Div([
            html.H3("Investment Return Parameters"),
            
            html.Label("Stock X Expected Return"),
            dcc.Slider(id="stock-return-slider", min=0.05, max=0.3, step=0.01, value=0.12, 
                      marks={0.05: '5%', 0.1: '10%', 0.2: '20%', 0.3: '30%'}),
            
            html.Label("Stock X Volatility"),
            dcc.Slider(id="stock-vol-slider", min=0.1, max=0.5, step=0.05, value=0.3, 
                      marks={0.1: '10%', 0.2: '20%', 0.3: '30%', 0.4: '40%', 0.5: '50%'}),
            
            html.Label("Index Expected Return"),
            dcc.Slider(id="index-return-slider", min=0.03, max=0.15, step=0.01, value=0.08, 
                      marks={0.05: '5%', 0.1: '10%', 0.15: '15%'}),
            
            html.Label("Index Volatility"),
            dcc.Slider(id="index-vol-slider", min=0.05, max=0.3, step=0.05, value=0.15, 
                      marks={0.05: '5%', 0.1: '10%', 0.2: '20%', 0.3: '30%'})
        ], className="six columns"),
        
        html.Div([
            html.H3("Simulation Controls"),
            
            html.Label("Random Factor"),
            dcc.Slider(id="random-slider", min=0, max=0.2, step=0.02, value=0.05,
                      marks={0: '0', 0.1: '0.1', 0.2: '0.2'}),
            
            html.Label("Number of Iterations"),
            dcc.Slider(id="iterations-slider", min=10, max=100, step=10, value=50,
                      marks={10: '10', 50: '50', 100: '100'}),
            
            html.Br(),
            html.Br(),
            
            html.Button("Run Simulation", id="run-button", className="button-primary", n_clicks=0),
            
            html.Div(id="bubble-indicator", style={'marginTop': 20, 'fontSize': 18})
        ], className="six columns")
    ], className="row"),
    
    # Results section
    html.Div([
        # Time series chart
        dcc.Graph(id="time-series-chart"),
        
        # Phase space chart
        dcc.Graph(id="phase-space-chart"),
    ]),
    
    # Store the intermediate data
    dcc.Store(id='simulation-data')
])

# Callback for running the simulation
@app.callback(
    Output('simulation-data', 'data'),
    Input('run-button', 'n_clicks'),
    [State('init-a-slider', 'value'),
     State('init-b-slider', 'value'),
     State('risk-a-slider', 'value'),
     State('risk-b-slider', 'value'),
     State('social-a-slider', 'value'),
     State('social-b-slider', 'value'),
     State('stock-return-slider', 'value'),
     State('stock-vol-slider', 'value'),
     State('index-return-slider', 'value'),
     State('index-vol-slider', 'value'),
     State('random-slider', 'value'),
     State('iterations-slider', 'value')]
)
def run_simulation(n_clicks, init_a, init_b, risk_a, risk_b, social_a, social_b, 
                  stock_return, stock_vol, index_return, index_vol, random_factor, iterations):
    if n_clicks == 0:
        # Return default empty data
        return None
    
    # Create and run the model
    model = InvestmentModel(
        initial_stock_x_investors_A=init_a,
        initial_stock_x_investors_B=init_b,
        risk_aversion_A=risk_a,
        risk_aversion_B=risk_b,
        social_influence_factor_A=social_a,
        social_influence_factor_B=social_b,
        stock_x_return_mean=stock_return,
        stock_x_return_std=stock_vol,
        index_return_mean=index_return,
        index_return_std=index_vol,
        randomness=random_factor,
        max_iterations=iterations
    )
    
    # Run simulation
    model.run_simulation()
    
    # Store results
    return {
        'history_A': model.history_A,
        'history_B': model.history_B,
        'bubble_formed': model.check_bubble_formation(threshold=0.7)
    }

# Callback for updating the time series chart
@app.callback(
    Output('time-series-chart', 'figure'),
    Input('simulation-data', 'data')
)
def update_time_series(data):
    if data is None:
        # Return empty figure
        return {
            'data': [],
            'layout': {
                'title': 'Investment Choice Over Time',
                'xaxis': {'title': 'Time Steps'},
                'yaxis': {'title': 'Fraction Investing in Stock X'}
            }
        }
    
    # Create the figure
    fig = {
        'data': [
            {'x': list(range(len(data['history_A']))), 'y': data['history_A'], 
             'type': 'line', 'name': 'Population A', 'line': {'color': 'blue'}},
            {'x': list(range(len(data['history_B']))), 'y': data['history_B'], 
             'type': 'line', 'name': 'Population B', 'line': {'color': 'green'}},
            {'x': list(range(len(data['history_A']))), 'y': [0.5] * len(data['history_A']), 
             'type': 'line', 'name': '50% Threshold', 'line': {'dash': 'dash', 'color': 'red'}}
        ],
        'layout': {
            'title': 'Investment Choice Over Time',
            'xaxis': {'title': 'Time Steps'},
            'yaxis': {'title': 'Fraction Investing in Stock X', 'range': [0, 1]},
            'legend': {'x': 0, 'y': 1}
        }
    }
    
    return fig

# Callback for updating the phase space chart
@app.callback(
    Output('phase-space-chart', 'figure'),
    Input('simulation-data', 'data')
)
def update_phase_space(data):
    if data is None:
        # Return empty figure
        return {
            'data': [],
            'layout': {
                'title': 'Phase Space: Population A vs Population B',
                'xaxis': {'title': 'Population A (Stock X Investors)'},
                'yaxis': {'title': 'Population B (Stock X Investors)'}
            }
        }
    
    # Create the figure
    fig = {
        'data': [
            {'x': data['history_A'], 'y': data['history_B'], 
             'type': 'scatter', 'mode': 'lines+markers', 'line': {'color': 'black'}},
            {'x': [data['history_A'][0]], 'y': [data['history_B'][0]], 
             'type': 'scatter', 'mode': 'markers', 'name': 'Start',
             'marker': {'size': 12, 'color': 'green'}},
            {'x': [data['history_A'][-1]], 'y': [data['history_B'][-1]], 
             'type': 'scatter', 'mode': 'markers', 'name': 'End',
             'marker': {'size': 12, 'color': 'red'}}
        ],
        'layout': {
            'title': 'Phase Space: Population A vs Population B',
            'xaxis': {'title': 'Population A (Stock X Investors)', 'range': [0, 1]},
            'yaxis': {'title': 'Population B (Stock X Investors)', 'range': [0, 1]},
            'legend': {'x': 0, 'y': 1}
        }
    }
    
    return fig

# Callback for updating the bubble indicator
@app.callback(
    Output('bubble-indicator', 'children'),
    Output('bubble-indicator', 'style'),
    Input('simulation-data', 'data')
)
def update_bubble_indicator(data):
    if data is None:
        return "Run simulation to check for bubble formation", {'marginTop': 20, 'fontSize': 18}
    
    if data['bubble_formed']:
        return "BUBBLE DETECTED", {'marginTop': 20, 'fontSize': 18, 'color': 'red', 'fontWeight': 'bold'}
    else:
        return "No bubble detected", {'marginTop': 20, 'fontSize': 18, 'color': 'green'}

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)