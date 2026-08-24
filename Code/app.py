import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Load the dataset
DATA_PATH = 'data/Processed_Students_Social_Media_Addiction.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Cannot find the file at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Convert column if temporal data is available
if 'Usage_Date' in df.columns:
    df['Usage_Date'] = pd.to_datetime(df['Usage_Date'], errors='coerce')

# Dropdown filter options
genders = sorted(df['Gender'].dropna().unique())
age_groups = sorted(df['Age_Group'].dropna().unique())
addiction_levels = sorted(df['Addiction_Level'].dropna().unique())
academic_levels = sorted(df['Academic_Level'].dropna().unique())

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# App Layout
app.layout = html.Div([
    html.H1("📊 Social Media Addiction Analytics Dashboard", style={'textAlign': 'center', 'marginBottom': 20}),

    dbc.Row([
        dbc.Col([
            html.H5("Filter by:"),
            html.Label("Gender"),
            dcc.Dropdown(
                id='gender-filter',
                options=[{'label': g, 'value': g} for g in genders],
                value=genders, multi=True
            ),
            html.Label("Age Group"),
            dcc.Dropdown(
                id='age-filter',
                options=[{'label': a, 'value': a} for a in age_groups],
                value=age_groups, multi=True
            ),
            html.Label("Addiction Level"),
            dcc.Dropdown(
                id='addiction-filter',
                options=[{'label': l, 'value': l} for l in addiction_levels],
                value=addiction_levels, multi=True
            ),
            html.Label("Academic Level"),
            dcc.Dropdown(
                id='academic-filter',
                options=[{'label': al, 'value': al} for al in academic_levels],
                value=academic_levels, multi=True
            ),
        ], width=3, style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'}),

        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id='platform-usage'), width=6),
                dbc.Col(dcc.Graph(id='addiction-distribution'), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='addiction-vs-academic'), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='sleep-by-addiction'), width=6),
                dbc.Col(dcc.Graph(id='mentalhealth-by-addiction'), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='cluster-plot'), width=12)
            ])
        ], width=9)
    ], style={'margin': '0px 20px'})
])

# Callback to update all graphs
@app.callback(
    Output('platform-usage', 'figure'),
    Output('addiction-distribution', 'figure'),
    Output('addiction-vs-academic', 'figure'),
    Output('sleep-by-addiction', 'figure'),
    Output('mentalhealth-by-addiction', 'figure'),
    Output('cluster-plot', 'figure'),
    Input('gender-filter', 'value'),
    Input('age-filter', 'value'),
    Input('addiction-filter', 'value'),
    Input('academic-filter', 'value')
)
def update_graphs(selected_genders, selected_ages, selected_addiction, selected_academic):
    filtered = df[
        df['Gender'].isin(selected_genders) &
        df['Age_Group'].isin(selected_ages) &
        df['Addiction_Level'].isin(selected_addiction) &
        df['Academic_Level'].isin(selected_academic)
    ]

    def empty_fig(title):
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(
                text="No data available for selected filters.",
                x=0.5, y=0.5, showarrow=False, font_size=16
            )]
        )
        return fig

    if filtered.empty:
        return (
            empty_fig("Most Used Social Media Platforms"),
            empty_fig("Addiction Score Distribution by Gender"),
            empty_fig("Addiction Score vs Academic Performance"),
            empty_fig("Sleep Hours by Addiction Level and Gender"),
            empty_fig("Mental Health Score by Addiction Level and Gender"),
            empty_fig("Cluster Grouping of Students")
        )

    # 1. Pie Chart
    fig1 = px.pie(filtered, names='Most_Used_Platform', title="Most Used Social Media Platforms") \
        if 'Most_Used_Platform' in filtered.columns else empty_fig("Most Used Social Media Platforms")

    # 2. Histogram
    fig2 = px.histogram(filtered, x='Addicted_Score', color='Gender', nbins=20,
                        title='Addiction Score Distribution by Gender') \
        if 'Addicted_Score' in filtered.columns else empty_fig("Addiction Score Distribution by Gender")

    # 3. Scatter Plot
    if 'Affects_Academic_Performance' in filtered.columns and 'Addicted_Score' in filtered.columns:
        filtered['Affects_Academic_Performance_Numeric'] = filtered['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0})
        academic_data = filtered.dropna(subset=['Affects_Academic_Performance_Numeric'])
        if len(academic_data) >= 2:
            fig3 = px.scatter(academic_data, x='Addicted_Score', y='Affects_Academic_Performance_Numeric',
                              color='Gender', facet_col='Academic_Level', trendline='ols',
                              labels={'Affects_Academic_Performance_Numeric': 'Affects Academics (1=Yes, 0=No)'},
                              title='Addiction Score vs Academic Performance')
        else:
            fig3 = empty_fig("Addiction Score vs Academic Performance")
    else:
        fig3 = empty_fig("Addiction Score vs Academic Performance")

    # 4. Box Plot: Sleep
    fig4 = px.box(filtered, x='Addiction_Level', y='Sleep_Hours_Per_Night', color='Gender',
                  title='Sleep Hours by Addiction Level and Gender') \
        if 'Sleep_Hours_Per_Night' in filtered.columns else empty_fig("Sleep Hours by Addiction Level and Gender")

    # 5. Box Plot: Mental Health
    fig5 = px.box(filtered, x='Addiction_Level', y='Mental_Health_Score', color='Gender',
                  title='Mental Health Score by Addiction Level and Gender') \
        if 'Mental_Health_Score' in filtered.columns else empty_fig("Mental Health Score by Addiction Level and Gender")

    # 6. 3D Cluster Plot
    cluster_cols = ['Addicted_Score', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']
    if all(col in filtered.columns for col in cluster_cols):
        cluster_data = filtered[cluster_cols].dropna()
        if len(cluster_data) >= 3:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(cluster_data)
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(scaled)
            cluster_data['Cluster'] = labels
            fig6 = px.scatter_3d(cluster_data, x='Addicted_Score', y='Avg_Daily_Usage_Hours',
                                 z='Sleep_Hours_Per_Night', color='Cluster',
                                 title='Cluster Grouping of Students')
        else:
            fig6 = empty_fig("Cluster Grouping of Students")
    else:
        fig6 = empty_fig("Cluster Grouping of Students")

    return fig1, fig2, fig3, fig4, fig5, fig6

# Run the Dash server
if __name__ == '__main__':
    app.run(debug=True)
