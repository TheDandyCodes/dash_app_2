import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import dash
import dash_bootstrap_components as dbc
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

liq_spread_table = pd.read_csv('LS.csv')
fix_params = pd.read_csv('params.csv')
df = pd.read_csv('contratos_BBDD.csv')

def real_balance_calc(tasa_interes_anual, num_pagos, monto_prestamo, cancelacion=0):

    if tasa_interes_anual==0.0:
        tasa_interes_anual = 0.1**7
    i = tasa_interes_anual/12
    nr = num_pagos - cancelacion
    ip = monto_prestamo
    mp = (ip * i) / (1 - (1 + i)**-num_pagos)
    A = ((1+i)**nr -1)
    B = (ip-(mp/i))
    C = A*B/nr
    return i**-1*(C+mp)

def duration_2(n_pagos, monto, tin, t):
    if tin==0:
        tin = 0.1**7
    npv = (monto * tin/12) / (1 - (1 + tin/12)**-n_pagos) / ((1+tin/12)**t)
    return np.sum(t*npv) / monto

def liquidity_spread_calc(liq_spread_table, duration):

    if len(liq_spread_table[(liq_spread_table.Time<=duration)].values) == 0:
        prev_ls_term = liq_spread_table.iloc[0:].LS.values[0]
        next_ls_term = prev_ls_term
    else:
        prev_time_term = liq_spread_table[(liq_spread_table.Time<=duration)].Time[-1:].values[0]

        if prev_time_term == liq_spread_table.iloc[-1:].Time.values[0]:
            prev_ls_term = liq_spread_table.iloc[-1:].LS.values[0]
            next_ls_term = prev_ls_term

        else:
            next_time_term = liq_spread_table.iloc[liq_spread_table[(liq_spread_table.Time<=duration)].index + 1, :].Time[-1:].values[0]
            prev_ls_term = liq_spread_table[liq_spread_table.Time == prev_time_term].LS.values[0]
            next_ls_term = liq_spread_table[liq_spread_table.Time == next_time_term].LS.values[0]

    if next_ls_term != prev_ls_term:
        return ((duration - prev_time_term) * (next_ls_term-prev_ls_term)/(next_time_term-prev_time_term)) + prev_ls_term
    else:
        return next_ls_term

def frenchAmortizationCalculator(monto_prestamo, tasa_interes_anual=0, num_pagos=1, cancelacion=0):

    # params = {'Monto':monto_prestamo, 'TIN':tasa_interes_anual, 'Pagos':num_pagos, 'Cancelacion':cancelacion, 'Comision_pagada': comision_pagada, 'Comision_apertura':comision_apertura, 'ITR':ITR, 'other':other, 'op_exp':op_exp, 'EL':EL, 'RWA':RWA}

    if tasa_interes_anual == 0:
        tasa_interes_anual = 0.1**7

    tasa_interes_mensual = tasa_interes_anual/12
    #cuota_fija = (monto_prestamo*tasa_interes_mensual*(1+tasa_interes_mensual)**(plazo*pagos_anuales))/((1+tasa_interes_mensual)**(plazo*pagos_anuales)-1)
    cuota_fija = (monto_prestamo * tasa_interes_mensual) / (1 - (1 + tasa_interes_mensual)**-num_pagos)
    saldo_final = monto_prestamo

    amortizaciones = []

    if cancelacion > 0:
        for i in range(1, (num_pagos-cancelacion)+1):
            saldo_inicial = saldo_final
            interes = tasa_interes_mensual*saldo_inicial
            amortizacion = cuota_fija-interes
            saldo_inicial = saldo_final
            if i==num_pagos-cancelacion:
                saldo_final = 0
            else:
                saldo_final -= amortizacion
            row = np.round([i, saldo_inicial, interes, amortizacion, cuota_fija, saldo_final], 3)
            amortizaciones.append(row)
        t = np.arange(1, (num_pagos-cancelacion)+1)

    else:
        for i in range(1, num_pagos+1):
            saldo_inicial = saldo_final
            interes = tasa_interes_mensual*saldo_inicial
            amortizacion = cuota_fija-interes
            saldo_inicial = saldo_final
            saldo_final -= amortizacion
            row = np.round([i, saldo_inicial, interes, amortizacion, cuota_fija, saldo_final], 3)
            amortizaciones.append(row)
        t = np.arange(1, num_pagos+1)

    duration = duration_2(num_pagos, monto_prestamo, tasa_interes_anual, t)

    cuadro = pd.DataFrame(amortizaciones, columns=['Mes', 'Saldo inicial', 'Interes', 'Principal', 'Cuota Fija', 'Saldo Final'])
    
    # li = cuadro['Saldo Final'][:-1].tolist()
    # li.append(monto_prestamo)
    # real_balance = np.mean(li)

    real_balance = real_balance_calc(tasa_interes_anual, num_pagos, monto_prestamo, cancelacion)
    total = pd.DataFrame({'Interes':sum(cuadro['Interes']), 'Principal':sum(cuadro['Principal'])+(cuadro['Saldo inicial'][-1:] - cuadro.Principal[-1:]).values[0], 'Total cuotas':(sum(cuadro['Interes'])+sum(cuadro['Principal'])+(cuadro['Saldo inicial'][-1:] - cuadro.Principal[-1:]).values[0]), 'Real Balance':real_balance, 'Duration':duration, 'Real term':num_pagos-cancelacion}, index=['Total'])
    

    return np.round(cuadro,2), np.round(total,4)

def getIndicators(monto_prestamo, tasa_interes_anual, num_pagos, cancelacion=0, comision_terceros=0, comision_rappels=0, comision_apertura=0, ITR=0, other=0, op_exp=0, EL=0, RWA=1):
    
    t = np.arange(1, (num_pagos-cancelacion)+1)
    duration = duration_2(num_pagos, monto_prestamo, tasa_interes_anual, t)
    real_balance = real_balance_calc(tasa_interes_anual, num_pagos, monto_prestamo, cancelacion)
    fees_collected = comision_apertura*12*(real_balance*(num_pagos-cancelacion))**-1
    fees_terceros = comision_terceros*12*(real_balance*(num_pagos-cancelacion))**-1
    fees_rappels = comision_rappels*12*(real_balance*(num_pagos-cancelacion))**-1
    fees_paid = fees_terceros+fees_rappels
    financial_fees = fees_collected-fees_paid
    
    # other = other*12*(real_balance*(num_pagos-cancelacion))**-1
    # op_exp = op_exp*12*(real_balance*(num_pagos-cancelacion))**-1
    LS = liquidity_spread_calc(liq_spread_table, duration) / 10**4
    TII = tasa_interes_anual + financial_fees
    TIE = LS + ITR
    NII = TII - TIE
    GM = NII - other
    NOI = GM - op_exp
    PBT = NOI - EL
    PAT = 0.715*PBT
    RORWA = PAT*real_balance / RWA

    # Añadir calculo TII, TIE y NII para hayar finalmente PBT y PAT, que es lo que se va a mostrar
    indicators = pd.DataFrame({'LS':LS, 'PBT':PBT, 'PAT':PAT, 'RORWA':RORWA}, index=['RORWA'])
    return np.round(indicators, 4)

def RORWA_flag(actual_rorwa):
    flag = 'Green'
    if actual_rorwa >= 2: flag = 'Green'
    elif actual_rorwa >= 1 and actual_rorwa<2: flag = 'Orange'
    else: flag = 'Red'
    return flag

def full_data(data):
    data['Duration'] = 0.0
    data['PAT'] = 0.0
    data['PBT'] = 0.0
    data['RORWA'] = 0.0
    data['LS'] = 0.0
    data['TII'] = 0.0
    data['TIE'] = 0.0
    data['NII'] = 0.0
    data['GM'] = 0.0
    data['NOI'] = 0.0
    data['Real term'] = data['Pagos'] - data['Cancelacion']
    for i in range(data.shape[0]):
        data.at[i, 'Real balance'] = real_balance_calc(data.loc[i, 'TIN'], data.loc[i, 'Pagos'], data.loc[i, 'Monto'], data.loc[i, 'Cancelacion'])
        data.at[i, 'Duration'] = duration_2(data.loc[i, 'Pagos'], data.loc[i, 'Monto'], data.loc[i, 'TIN'], np.arange(1, (data.loc[i, 'Pagos']-data.loc[i, 'Cancelacion'])+1))
        data.at[i, 'LS'] = liquidity_spread_calc(liq_spread_table, data.at[i, 'Duration']) / 10**4
    data['Real term Ponderado'] = data['Real term']*data['Real balance']
    data.fees_apertura = (data.fees_apertura*12)/data['Real term Ponderado']
    data.fees_terceros = (data.fees_terceros*12)/data['Real term Ponderado']
    data.fees_rappels = (data.fees_rappels*12)/data['Real term Ponderado']
    data.fin_fees = data.fees_apertura - data.fees_terceros - data.fees_rappels
    data.TII = data.TIN + data.fin_fees
    data.TIE = data.ITR+ data.LS
    data.NII = data.TII - data.TIE
    data.GM = data.NII - data.non_fin_fees
    data.NOI = data.GM - data.op_exp
    data.PBT = data.NOI - data.EL
    data.PAT = data.PBT * 0.715
    data.RORWA = data.PAT*data['Monto'] / data.RWA
    data.Mes = [datetime.date(2023, mes, 1) for mes in data.Mes]
    data.Mes = pd.to_datetime(data.Mes)
    data.drop(data[data['Real balance'].isnull()].index, inplace=True)
    return data
    
def line_plot(pivot_table):
    fig = go.Figure()

    # Crear y personalizar las trazas
    fig.add_trace(go.Scatter(x=pivot_table.index, y=pivot_table.ITR, name='ITR',
                            line=dict(color='rgb(102,124,165)', width=3),
                            mode='lines+markers+text', text=pivot_table.ITR,
                            textposition='top center',
                            textfont=dict(size=10, color='rgb(102,124,165)'),
                            marker=dict(size=8),
                            texttemplate='%{y:.2f}%',
                            # line_shape='spline' 
                            ))

    fig.add_trace(go.Scatter(x=pivot_table.index, y=pivot_table.LS, name='LS',
                            line=dict(color='rgb(209,190,102)', width=3),
                            mode='lines+markers+text', text=pivot_table.LS,
                            textposition='top center',
                            textfont=dict(size=10, color='rgb(209,190,102)'),
                            marker=dict(size=8),
                            texttemplate='%{y:.2f}%',
                            # line_shape='spline'  
                            ))

    fig.add_trace(go.Scatter(x=pivot_table.index, y=pivot_table.NII, name='NII',
                            line=dict(color='rgb(0,26,72)', width=3),
                            mode='lines+markers+text', text=pivot_table.NII,
                            textposition='top center',
                            textfont=dict(size=10, color='rgb(0,26,72)'),
                            marker=dict(size=8),
                            texttemplate='%{y:.2f}%',
                            # line_shape='spline'  
                            ))

    fig.add_trace(go.Scatter(x=pivot_table.index, y=pivot_table.TII, name='TII',
                            line=dict(color='rgb(178,178,178)', width=3),
                            mode='lines+markers+text', text=pivot_table.TII,
                            textposition='top center',
                            textfont=dict(size=10, color='rgb(178,178,178)'),
                            marker=dict(size=8),
                            texttemplate='%{y:.2f}%',
                            # line_shape='spline'  
                            ))

    # Editar el diseño
    fig.update_layout(title='Evolución por mes del producto',
                    title_font=dict(size=18, color='rgb(0,26,72)', family= 'sans-serif'),
                    xaxis_title='Month',
                    yaxis_title='%', 
                    height=450,
                    width=500,
                    plot_bgcolor='white',
                    #yaxis=dict(range=[-1.5, 4]),
                    xaxis=dict(showline=True, linecolor='rgb(184,184,184)', linewidth=2),
                    yaxis=dict(showline=True, linecolor='rgb(184,184,184)', linewidth=2),
                    transition=dict(duration=1000, ordering='traces first')
                    )
    return fig


def bubble_chart(pivot_table):
    fig = px.scatter(pivot_table, x=pivot_table.RORWA, y=pivot_table.TIN,
                size=pivot_table.Monto/100, color=pivot_table.index,
                    hover_name=pivot_table.index, text=pivot_table.index, size_max=90, width=700)
    fig.add_shape(type='line',
                x0=2, y0=0, x1=2, y1=1, xref='x', yref='paper',
                line=dict(color='rgb(255, 105, 105)', width=2, dash='dash'))

    fig.update_layout(title='Brand Scatterplot by Volume',
                        xaxis_title='RORWA %',
                        yaxis_title='TIN %',
                        title_font=dict(size=18, color='rgb(0,26,72)', family= 'sans-serif'), 
                        height=450,
                        width=500,
                        plot_bgcolor='white',
                        #yaxis=dict(range=[-1.5, 4]),
                        xaxis=dict(showline=True, linecolor='rgb(184,184,184)', linewidth=2),
                        yaxis=dict(showline=True, linecolor='rgb(184,184,184)', linewidth=2),
                        transition=dict(duration=1000, ordering='traces first')
                        )

    return fig

def make_pandl(data, num_productos, product1='', product2='', product3=''):

    mes = np.round(data.pivot_table(values=['TIN', 'fees_terceros', 'fees_rappels', 'fees_apertura', 'TII', 'TIE', 'NII', 'non_fin_fees', 'GM', 'op_exp', 'NOI', 'EL', 'PBT', 'PAT', 'RWA', 'RORWA'],
                       index='Mes',
                       aggfunc=lambda rows: np.average(rows, weights=data.loc[rows.index, 'Real term Ponderado']))*100, 3).reset_index().Mes


    negs = ['fees_terceros', 'fees_rappels', 'non_fin_fees', 'op_exp', 'TIE', 'EL']
    year = data['Mes'].dt.strftime('%Y')[0]

    if num_productos == 1:
        data_pl = np.round(data[data.Cluster == product1].pivot_table(values=['Monto', 'TIN', 'fees_terceros', 'fees_rappels', 'fees_apertura', 'TII', 'TIE', 'NII', 'non_fin_fees', 'GM', 'op_exp', 'NOI', 'EL', 'PBT', 'PAT', 'RWA', 'RORWA'],
                            index='Mes',
                            aggfunc=lambda rows: np.average(rows, weights=data.loc[rows.index, 'Real term Ponderado'])), 3).reset_index(drop=True)
        data_pl[negs] = data_pl[negs]*-1
    elif num_productos == 2:
        data_pl = np.round(data[(data.Cluster == product1) | (data.Cluster == product2)].pivot_table(values=['Monto', 'TIN', 'fees_terceros', 'fees_rappels', 'fees_apertura', 'TII', 'TIE', 'NII', 'non_fin_fees', 'GM', 'op_exp', 'NOI', 'EL', 'PBT', 'PAT', 'RWA', 'RORWA'],
                            index='Mes',
                            aggfunc=lambda rows: np.average(rows, weights=data.loc[rows.index, 'Real term Ponderado'])), 3).reset_index(drop=True)
        data_pl[negs] = data_pl[negs]*-1
    elif num_productos == 3:
        data_pl = np.round(data[(data.Cluster == product1) | (data.Cluster == product2) | (data.Cluster == product3)].pivot_table(values=['Monto', 'TIN', 'fees_terceros', 'fees_rappels', 'fees_apertura', 'TII', 'TIE', 'NII', 'non_fin_fees', 'GM', 'op_exp', 'NOI', 'EL', 'PBT', 'PAT', 'RWA', 'RORWA'],
                            index='Mes',
                            aggfunc=lambda rows: np.average(rows, weights=data.loc[rows.index, 'Real term Ponderado'])), 3).reset_index(drop=True)
        data_pl[negs] = data_pl[negs]*-1
    else:
        data_pl = np.round(data.pivot_table(values=['Monto', 'TIN', 'fees_terceros', 'fees_rappels', 'fees_apertura', 'TII', 'TIE', 'NII', 'non_fin_fees', 'GM', 'op_exp', 'NOI', 'EL', 'PBT', 'PAT', 'RWA', 'RORWA'],
                            index='Mes',
                            aggfunc=lambda rows: np.average(rows, weights=data.loc[rows.index, 'Real term Ponderado'])), 3).reset_index(drop=True)
        data_pl[negs] = data_pl[negs]*-1
        
    data_pl = data_pl.reindex(columns=['Monto', 'TIN', 'fees_terceros', 'fees_rappels', 'fees_apertura', 'TII', 'TIE', 'NII', 'non_fin_fees', 'GM', 'op_exp', 'NOI', 'EL', 'PBT', 'PAT', 'RWA', 'RORWA'])
    data_pl[['TIN', 'fees_terceros', 'fees_rappels', 'fees_apertura', 'TII', 'TIE', 'NII', 'non_fin_fees', 'GM', 'op_exp', 'NOI', 'EL', 'PBT', 'PAT', 'RORWA']] = np.round(data_pl[['TIN', 'fees_terceros', 'fees_rappels', 'fees_apertura', 'TII', 'TIE', 'NII', 'non_fin_fees', 'GM', 'op_exp', 'NOI', 'EL', 'PBT', 'PAT', 'RORWA']]*100, 3)
    data_pl.insert(0, 'Mes', mes)
    data_pl = data_pl.T
    data_pl.insert(0, 'P&L NB', ['Mes', 'Monto', 'TIN', 'fees_terceros', 'fees_rappels', 'fees_apertura', 'TII', 'TIE', 'NII', 'non_fin_fees', 'GM', 'op_exp', 'NOI', 'EL', 'PBT', 'PAT', 'RWA', 'RORWA'])
    data_pl.columns = ['P&L NB']+['Enero '+year, 'Febrero '+year, 'Marzo '+year, 'Abril '+year, 'Mayo '+year, 'Junio '+year, 'Julio '+year, 'Agosto '+year, 'Septiembre '+year, 'Octubre '+year, 'Noviembre '+year, 'Diciembre '+year]
    data_pl = data_pl[['P&L NB'] + list(data_pl.columns[::-1])].iloc[:,:-1]
    data_pl = data_pl.tail(17)
    return data_pl

df = full_data(df)
df_pl = make_pandl(df, 4)

############################### DASH ##############################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'styles.css'])
server = app.server

df = pd.read_csv('pricing_parameters/contratos_BBDD.csv')

opciones_dropdown_1 = [
    {'label': 'Consumo', 'value': 'consumo'},
    {'label': 'Auto Usado', 'value': 'auto_usado'},
    {'label': 'Auto Nuevo', 'value': 'auto_nuevo'},
    {'label': 'Directo', 'value': 'directo'},
]

opciones_dropdown_2 = [
    {'label': 'Suc A', 'value': 'suc_a'},
    {'label': 'Suc B', 'value': 'suc_b'},
    {'label': 'Suc C', 'value': 'suc_c'},
    {'label': 'Suc D', 'value': 'suc_d'},
]

opciones_dropdown_3 = [
    {'label': 'Marca A', 'value': 'marca_a'},
    {'label': 'Marca B', 'value': 'marca_b'},
    {'label': 'Marca C', 'value': 'marca_c'},
    {'label': 'Marca D', 'value': 'marca_d'},
]

filas_gris = [7, 9, 11]
filas_rojas = [13, 14, 16]

type_tabla = [{'if': {'column_id': c},
               'font-family': 'sans-serif'} for c in df_pl.columns]

dropdown1_card_content = [
    dbc.CardHeader("Tipo de producto", style={'background-color': 'rgb(0, 26, 72)', 'color':'white', 'border':'none'}),
    dbc.CardBody(
        [
            dbc.Row([
                dcc.Dropdown(
                id='contrato-input',
                value=['consumo'],
                options=opciones_dropdown_1,
                # labelStyle={'display': 'block'},
                style={'margin': 'auto'},
                multi=True,
                ),
            ]),
        ]
    ),
]

card_dropdown_1 = dbc.Card(dropdown1_card_content, style={'background-color': 'white', 'margin': '10px', 'border':'none'})

dropdown2_card_content = [
    dbc.CardHeader("Sucursal", style={'background-color': 'rgb(0, 26, 72)', 'color':'white', 'border':'none'}),
    dbc.CardBody(
        [
            dbc.Row([
                dcc.Dropdown(
                id='contrato-input-2',
                value=[],
                options=opciones_dropdown_2,
                # labelStyle={'display': 'block'},
                style={'margin': 'auto'},
                multi=True,
                ),
            ]),
        ]
    ),
]

card_dropdown_2 = dbc.Card(dropdown2_card_content, style={'background-color': 'white', 'margin': '10px', 'border':'none'})

dropdown3_card_content = [
    dbc.CardHeader("Marca", style={'background-color': 'rgb(0, 26, 72)', 'color':'white', 'border':'none'}),
    dbc.CardBody(
        [
            dbc.Row([
                dcc.Dropdown(
                id='contrato-input-3',
                value=[],
                options=opciones_dropdown_3,
                # labelStyle={'display': 'block'},
                style={'margin': 'auto'},
                multi=True,
                ),
            ]),
        ]
    ),
]

card_dropdown_3 = dbc.Card(dropdown3_card_content, style={'background-color': 'white', 'margin': '10px', 'border':'none'})

# Contenido de la primera card
card_table_content = [
    # dbc.CardHeader("Card de Grafico 1"),
    dbc.CardBody(
        [
            dcc.Graph(id='line-1'),
        ]
    ),
]

# Definición de la primera card
card_table = dbc.Card(card_table_content, style={'background-color': 'white', 'margin': '10px', 'border':'none'})

# Contenido de la segunda card (Gráfico)
card_chart_content = [
    # dbc.CardHeader("Card de Gráfico 2"),
    dbc.CardBody(
        [
            dcc.Graph(id='line-2'),
        ]
    ),
]

# Definición de la segunda card
card_chart = dbc.Card(card_chart_content, style={'background-color': 'white', 'margin': '10px', 'border':'none'})

pandl_card_content = [
    # dbc.CardHeader("P&L"),
    dbc.CardBody(
        [
           dash_table.DataTable(id='pandl',
                                 style_data_conditional=[
            {
                'if': {'row_index': i},
                'backgroundColor': 'rgb(178,178,178)',
                'color': 'white',
                'font-weight': 'bold',
            }
            for i in filas_gris
        ] +[
            {
                'if': {'row_index': i},
                'backgroundColor': 'rgb(193,168,51)',
                'color': 'white',
                'font-weight': 'bold',
            }
            for i in filas_rojas
        ]+[

            {
                'if': {'column_id': 'Inputs'},
                'font-weight': 'bold',
            }
        ],

        style_header={
            'backgroundColor': 'rgb(0,26,72)',
            'fontWeight': 'bold',
            'color': 'white'
        },
                                 
        page_size=17, style_table={'overflowX': 'auto', 'border': '1px solid white'})
        ]
    ),
]

card_pandl = dbc.Card(pandl_card_content, style={'background-color': 'white', 'margin': '10px 30px', 'border':'none'})

card_content_inv = [
    dbc.CardBody(
        [
            html.P("Inversion", style={'text-align': 'center', 'font-weight': 'bold', 'color':'rgb(0,26,72)'}, className="card-title"),
            html.P(id='valor-inversion', style={'font-size':'14px', 'text-align': 'center', 'color':'rgb(0,26,72)'}),
        ]
    )
]

card_content_rorwa = [
    dbc.CardBody(
        [
            html.P("RORWA", style={'text-align': 'center', 'font-weight':'bold', 'color':'rgb(0,26,72)'}, className="card-title"),
            html.P(id='valor-td', style={'font-size':'14px', 'text-align': 'center'}),
        ]
    )
]

card_content_tin = [
    dbc.CardBody(
        [
            html.P("TIN", style={'text-align': 'center', 'font-weight':'bold', 'color':'rgb(0,26,72)'}, className="card-title"),
            html.P(id='valor-tin', style={'font-size':'14px', 'text-align': 'center', 'color':'rgb(0,26,72)'}),
        ]
    )
]

card_content_nii = [
    dbc.CardBody(
        [
            html.P("Net Int. Inc.", style={'text-align': 'center', 'font-weight':'bold','color':'rgb(0,26,72)'}, className="card-title"),
            html.P(id='valor-nii', style={'font-size':'14px', 'text-align': 'center', 'color':'rgb(0,26,72)'}),
        ]
    )
]

card_content_finfee = [
    dbc.CardBody(
        [
            html.P("Fin. fees", style={'text-align': 'center', 'font-weight':'bold', 'color':'rgb(0,26,72)'}, className="card-title"),
            html.P(id='valor-finfee', style={'font-size':'14px', 'text-align': 'center', 'color':'rgb(0,26,72)'}),
        ]
    )
]

card_content_gm = [
    dbc.CardBody(
        [
            html.P("Gross Marg.", style={'text-align': 'center', 'font-weight':'bold', 'color':'rgb(0,26,72)'}, className="card-title"),
            html.P(id='valor-gm', style={'font-size':'14px', 'text-align': 'center', 'color':'rgb(0,26,72)'}),
        ]
    )
]

card_content_pat = [
    dbc.CardBody(
        [
            html.P("PAT", style={'text-align': 'center', 'font-weight':'bold', 'color':'rgb(0,26,72)'}, className="card-title"),
            html.P(id='valor-pat', style={'font-size':'12px', 'text-align': 'center', 'color':'rgb(0,26,72)'}),
        ]
    )
]

kpi_cards = [card_content_inv, card_content_rorwa, card_content_tin, card_content_nii, card_content_finfee, card_content_gm, card_content_pat]

# Crear las cards internas y colocarlas en las columnas
cards = [dbc.Col(dbc.Card(i, color="rgb(204, 211, 225)", outline=True, style={'width': '130px', 'height': '80px', 'border':'none'}), className='my-3') for i in kpi_cards]

# Crear la fila con las cards internas
card_row = dbc.Row(cards, justify="center")

# Crear la card principal que contiene la fila con las cards internas
main_card_content = dbc.CardBody([card_row])
kpi_card = dbc.Card(main_card_content, color="white", outline=True, style={'margin':'10px 30px'})

# Diseño del layout
app.layout = html.Div([
    dbc.Row([
        # Producto
        dbc.Col(card_dropdown_1, width=4),
        # Sucursal
        dbc.Col(card_dropdown_2, width=4),
        # Marca
        dbc.Col(card_dropdown_3, width=4),
    ], className="m-2"),
    kpi_card,
    dbc.Row([
        # Primera card (Tabla)
        dbc.Col(card_table, width=6),

        # Segunda card (Gráfico)
        dbc.Col(card_chart, width=6),
    ], className="m-2"),
    card_pandl

    # Card contenedora de KPIs
], style={'margin': '50px'})



#Función de callback para crear un DataFrame y actualizar la visualización
@app.callback(
    [Output('line-1', 'figure'), Output('line-2', 'figure'), Output('pandl', 'data')],#, Output('monto-value', 'children'), Output('rorwa-value', 'children'), Output('tin-value', 'children'), Output('nii-value', 'children'), Output('finfees-value', 'children'), Output('gm-value', 'children'), Output('pat-value', 'children')],
    [Input('contrato-input', 'value')],
)
def table(valor8):

    # pivot_rorwa = np.round(df.pivot_table(values=['RORWA', 'TIN', 'Monto', 'NII', 'fees_terceros', 'fees_rappels', 'GM', 'PAT'],
    #                    index='Cluster',
    #                    aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'Real term Ponderado']))*100, 3)

    if len(valor8) == 0:
        return _
    elif len(valor8) == 1:
        pivot = np.round(df[df.Cluster==valor8[0]].pivot_table(values=['ITR', 'NII', 'TII', 'LS'],
                        index='Mes',
                        aggfunc=lambda rows: np.average(rows, weights=df[df.Cluster==valor8[0]].loc[rows.index, 'Real term Ponderado']))*100, 3)
        pivot_rorwa = np.round(df[df.Cluster==valor8[0]].pivot_table(values=['RORWA', 'TIN', 'Monto', 'NII', 'fees_terceros', 'fees_rappels', 'GM', 'PAT'],
                       index='Cluster',
                       aggfunc=lambda rows: np.average(rows, weights=df[df.Cluster==valor8[0]].loc[rows.index, 'Real term Ponderado']))*100, 3)
        pandl_table = make_pandl(df, 1, valor8[0])
        
    elif len(valor8) == 2:
        pivot = np.round(df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1])].pivot_table(values=['ITR', 'NII', 'TII', 'LS'],
                        index='Mes',
                        aggfunc=lambda rows: np.average(rows, weights=df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1])].loc[rows.index, 'Real term Ponderado']))*100, 3)
        pivot_rorwa = np.round(df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1])].pivot_table(values=['RORWA', 'TIN', 'Monto', 'NII', 'fees_terceros', 'fees_rappels', 'GM', 'PAT'],
                       index='Cluster',
                       aggfunc=lambda rows: np.average(rows, weights=df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1])].loc[rows.index, 'Real term Ponderado']))*100, 3)
        pandl_table = make_pandl(df, 2, valor8[0], valor8[1])

    elif len(valor8) == 3:
        pivot = np.round(df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1]) | (df.Cluster==valor8[2])].pivot_table(values=['ITR', 'NII', 'TII', 'LS'],
                        index='Mes',
                        aggfunc=lambda rows: np.average(rows, weights=df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1]) | (df.Cluster==valor8[2])].loc[rows.index, 'Real term Ponderado']))*100, 3)
        pivot_rorwa = np.round(df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1]) | (df.Cluster==valor8[2])].pivot_table(values=['RORWA', 'TIN', 'Monto', 'NII', 'fees_terceros', 'fees_rappels', 'GM', 'PAT'],
                       index='Cluster',
                       aggfunc=lambda rows: np.average(rows, weights=df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1]) | (df.Cluster==valor8[2])].loc[rows.index, 'Real term Ponderado']))*100, 3)
        pandl_table = make_pandl(df, 3, valor8[0], valor8[1], valor8[2])
    else:
        pivot = np.round(df.pivot_table(values=['ITR', 'NII', 'TII', 'LS'],
                        index='Mes',
                        aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'Real term Ponderado']))*100, 3)
        pivot_rorwa = np.round(df.pivot_table(values=['RORWA', 'TIN', 'Monto', 'NII', 'fees_terceros', 'fees_rappels', 'GM', 'PAT'],
                       index='Cluster',
                       aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'Real term Ponderado']))*100, 3)
        pandl_table = make_pandl(df, 4)
    

    fig1 = line_plot(pivot, ", ".join(valor8))
    fig2 = bubble_chart(pivot_rorwa)
    

    return fig1, fig2, pandl_table.to_dict('records')

@app.callback(
    [Output('valor-inversion', 'children'), Output('valor-td', 'children'), Output('valor-tin', 'children'), Output('valor-nii', 'children'), Output('valor-finfee', 'children'), Output('valor-gm', 'children'), Output('valor-pat', 'children')],
    [Input('contrato-input', 'value')],
)
def valores(valor8):

    if len(valor8) == 0:
        return _
    elif len(valor8) == 1:
        pivot_rorwa = np.round(df[df.Cluster==valor8[0]].pivot_table(values=['RORWA', 'TIN', 'Monto', 'NII', 'fees_terceros', 'fees_rappels', 'GM', 'PAT'],
                       index='Cluster',
                       aggfunc=lambda rows: np.average(rows, weights=df[df.Cluster==valor8[0]].loc[rows.index, 'Real term Ponderado']), margins=True)*100, 2)
        
    elif len(valor8) == 2:
        pivot_rorwa = np.round(df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1])].pivot_table(values=['RORWA', 'TIN', 'Monto', 'NII', 'fees_terceros', 'fees_rappels', 'GM', 'PAT'],
                       index='Cluster',
                       aggfunc=lambda rows: np.average(rows, weights=df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1])].loc[rows.index, 'Real term Ponderado']), margins=True)*100, 2)

    elif len(valor8) == 3:
        pivot_rorwa = np.round(df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1]) | (df.Cluster==valor8[2])].pivot_table(values=['RORWA', 'TIN', 'Monto', 'NII', 'fees_terceros', 'fees_rappels', 'GM', 'PAT'],
                       index='Cluster',
                       aggfunc=lambda rows: np.average(rows, weights=df[(df.Cluster==valor8[0]) | (df.Cluster==valor8[1]) | (df.Cluster==valor8[2])].loc[rows.index, 'Real term Ponderado']), margins=True)*100, 2)
    else:
        pivot_rorwa = np.round(df.pivot_table(values=['RORWA', 'TIN', 'Monto', 'NII', 'fees_terceros', 'fees_rappels', 'GM', 'PAT'],
                       index='Cluster',
                       aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'Real term Ponderado']), margins=True)*100, 2)

    m = str(pivot_rorwa.Monto['All']/100)
    r = pivot_rorwa.RORWA['All']
    color = 'green' if r > 2 else 'red'
    r_rag = html.Span(
        children=f'{r}',
        style={'color': color}
    )
    tin = str(pivot_rorwa.TIN['All'])
    nii = str(pivot_rorwa.NII['All'])
    finfee = str(np.round(pivot_rorwa.fees_terceros['All'] + pivot_rorwa.fees_rappels['All'], 2))
    gm = str(pivot_rorwa.GM['All'])
    pat = str(pivot_rorwa.PAT['All'])

    return m, r_rag, tin, nii, finfee, gm, pat


if __name__ == "__main__":
    app.run_server(debug=True, port=8070)

