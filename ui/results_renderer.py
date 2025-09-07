# ui/results_renderer.py
"""
Módulo dedicado exclusivamente a renderizar los resultados del backtest global en la interfaz de Streamlit.
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import calendar
import plotly.graph_objects as go
import plotly.express as px

from agent_core.metrics import calculate_performance_metrics

def generate_calendar_html(pnl_by_day, year, month, monthly_pnl, monthly_start_equity):
    """Genera el HTML para el calendario de rendimiento mensual."""
    month_name = datetime.date(year, month, 1).strftime('%B %Y')
    
    calendar.setfirstweekday(calendar.MONDAY)
    cal = calendar.monthcalendar(year, month)

    pnl_class = "pnl-win" if monthly_pnl >= 0 else "pnl-loss"
    
    percentage_change_str = ""
    if monthly_start_equity > 0:
        percentage_change = (monthly_pnl / monthly_start_equity) * 100
        percentage_change_str = f"({percentage_change:+.2f}%)"

    monthly_pnl_str = (
        f'<span class="{pnl_class}">'
        f'${monthly_pnl:,.2f} {percentage_change_str}'
        f'</span>'
    )

    html = f"""
    <style>
        .calendar {{ font-family: sans-serif; border-collapse: collapse; width: 100%; }}
        .calendar th {{ background-color: #333; color: white; text-align: center; padding: 10px; font-size: 14px; }}
        .calendar td {{ border: 1px solid #444; height: 90px; vertical-align: top; padding: 5px; width: 12.5%; }}
        .calendar .day-number {{ font-size: 12px; color: #aaa; text-align: left; }}
        .calendar .pnl-win {{ color: #26a69a; }}
        .calendar .pnl-loss {{ color: #ef5350; }}
        .daily-pnl {{ font-weight: bold; font-size: 16px; text-align: center; margin-top: 10px; }}
        .calendar .trade-count {{ font-size: 11px; color: #888; text-align: center; }}
        .calendar .empty-day {{ background-color: #2b2b2b; }}
        .weekly-summary-cell {{ vertical-align: middle !important; text-align: center; }}
        .week-label {{ font-size: 14px; font-weight: bold; color: #ccc; }}
        .week-pnl {{ font-size: 16px; }}
        .week-trades {{ font-size: 12px; color: #888; }}
        
        .monthly-summary .pnl-win {{ color: #26a69a; }}
        .monthly-summary .pnl-loss {{ color: #ef5350; }}
    </style>
    <h3 class="monthly-summary" style="text-align: center; font-size: 20px;">{month_name} {monthly_pnl_str}</h3>
    <table class="calendar">
        <tr><th>Lun</th><th>Mar</th><th>Mié</th><th>Jue</th><th>Vie</th><th>Sáb</th><th>Dom</th><th style="background-color: #1e1e1e;">Resumen Semanal</th></tr>
    """
    
    for i, week in enumerate(cal):
        html += "<tr>"
        week_pnl = 0
        week_trades = 0
        for day in week:
            if day == 0:
                html += '<td class="empty-day"></td>'
            else:
                date = datetime.date(year, month, day)
                day_data = pnl_by_day.get(date)
                if day_data:
                    pnl = day_data['pnl']
                    count = day_data['count']
                    week_pnl += pnl
                    week_trades += count
                    pnl_class = "pnl-win" if pnl >= 0 else "pnl-loss"
                    pnl_text = f"${pnl:,.2f}"
                    count_text = f"{int(count)} trade{'s' if count > 1 else ''}"
                    html += f'<td><div class="day-number">{day}</div><div class="daily-pnl {pnl_class}">{pnl_text}</div><div class="trade-count">{count_text}</div></td>'
                else:
                    html += f'<td><div class="day-number">{day}</div></td>'
        
        if week_trades > 0:
            pnl_class = "pnl-win" if week_pnl >= 0 else "pnl-loss"
            html += f'<td class="weekly-summary-cell">'
            html += f'<div class="week-label">Semana {i+1}</div>'
            html += f'<div class="week-pnl {pnl_class}">${week_pnl:,.2f}</div>'
            html += f'<div class="week-trades">{int(week_trades)} trades</div>'
            html += '</td>'
        else:
            html += '<td class="empty-day"></td>'
            
        html += "</tr>"
    html += "</table>"
    return html

def render_global_results(filter_name: str = ""):
    """Función principal para mostrar todos los componentes de la página de resultados."""
    st.header("Resultados del Backtest Global")
    
    trades_df_raw = st.session_state.session_trades
    equity_df = st.session_state.global_equity_history
    
    if trades_df_raw.empty:
        st.warning("La estrategia no generó ninguna operación en el período seleccionado.")
        return

    trades_df = trades_df_raw.copy()
    for col in ['entry_time', 'exit_time', 'pnl_net']:
        trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')
    trades_df.dropna(subset=['entry_time', 'exit_time', 'pnl_net'], inplace=True)
    
    trades_df['entry_time_dt'] = pd.to_datetime(trades_df['entry_time'], unit='s', utc=True).dt.tz_convert(st.session_state.ui_display_tz)

    metrics = calculate_performance_metrics(trades_df.to_dict('records'), st.session_state.ui_initial_capital, equity_df.values.tolist())

    st.markdown("---")
    
    st.markdown("""
    <style>
    .metric-container {
        border: 1px solid #444;
        border-radius: 8px;
        padding: 10px 15px;
        background-color: #222;
        text-align: center;
    }
    .metric-label {
        font-size: 14px;
        color: #aaa;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #fff;
        margin: 5px 0;
    }
    .metric-delta {
        font-size: 14px;
    }
    .metric-delta-positive {
        color: #26a69a;
    }
    .metric-delta-negative {
        color: #ef5350;
    }
    </style>
    """, unsafe_allow_html=True)

    def create_metric_html(label, value, delta=None, delta_color="positive"):
        delta_html = ""
        if delta:
            delta_class = "metric-delta-positive" if delta_color == "positive" else "metric-delta-negative"
            arrow = "▲" if delta_color == "positive" else "▼"
            delta_html = f'<div class="metric-delta {delta_class}">{arrow} {delta}</div>'
        
        return f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """

    col1, col2, col3, col4 = st.columns(4)

    ganancia_neta_pct = metrics.get('Ganancia Neta Total (%)', 0)
    delta_color = "positive" if ganancia_neta_pct >= 0 else "negative"
    
    with col1:
        st.markdown(create_metric_html(
            "Ganancia Neta Total",
            f"${metrics.get('Ganancia Neta Total ($)', 0):,.2f}",
            f"{ganancia_neta_pct:.2f}%",
            delta_color
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_html(
            "Trades Totales",
            f"{metrics.get('Total Trades', 0)}"
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_html(
            "Win Rate",
            f"{metrics.get('Win Rate (%)', 0):.2f}%"
        ), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_html(
            "Profit Factor",
            f"{metrics.get('Profit Factor', 'N/A')}"
        ), unsafe_allow_html=True)
    
    st.markdown("---")

    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.subheader("Curva de Equity")
        if not equity_df.empty:
            equity_df_chart = equity_df.copy()
            equity_df_chart['time'] = pd.to_datetime(equity_df_chart['time'], unit='s', utc=True)
            equity_df_chart = equity_df_chart.set_index('time')
            
            if equity_df_chart.index.duplicated().any():
                equity_df_chart = equity_df_chart[~equity_df_chart.index.duplicated(keep='last')]
            
            equity_df_chart = equity_df_chart.resample('D').last().ffill()
            
            fig = go.Figure()
            initial_capital = st.session_state.ui_initial_capital
            
            fig.add_trace(go.Scatter(
                x=equity_df_chart.index,
                y=[initial_capital] * len(equity_df_chart),
                mode='lines', line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=equity_df_chart.index,
                y=np.maximum(equity_df_chart['equity'], initial_capital),
                mode='lines', line=dict(width=0),
                fillcolor='rgba(44, 160, 44, 0.3)', fill='tonexty',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=equity_df_chart.index,
                y=np.minimum(equity_df_chart['equity'], initial_capital),
                mode='lines', line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=equity_df_chart.index,
                y=[initial_capital] * len(equity_df_chart),
                mode='lines', line=dict(width=0),
                fillcolor='rgba(239, 83, 80, 0.3)', fill='tonexty',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=equity_df_chart.index, 
                y=equity_df_chart['equity'],
                mode='lines', name='Equity', 
                line=dict(color='#5DADE2', width=2)
            ))
            fig.add_shape(type='line',
                x0=equity_df_chart.index.min(), y0=initial_capital,
                x1=equity_df_chart.index.max(), y1=initial_capital,
                line=dict(color='white', width=1, dash='dash')
            )

            fig.update_layout(
                template="plotly_dark",
                showlegend=False,
                yaxis_title="Capital ($)",
                xaxis_title="Fecha",
                hovermode="x unified"
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"{filter_name}_equity",
            )

    with chart_col2:
        st.subheader("Métricas Adicionales")
        avg_win = metrics.get('Ganancia Promedio ($)', 0)
        avg_loss = metrics.get('Pérdida Promedio ($)', 0)
        
        st.metric(
            "Trade Ganador Promedio",
            f"${avg_win:,.2f}",
            key=f"{filter_name}_metric_avg_win",
        )
        st.metric(
            "Trade Perdedor Promedio",
            f"${avg_loss:,.2f}",
            key=f"{filter_name}_metric_avg_loss",
        )
        st.metric(
            "Max Drawdown",
            f"{metrics.get('Max Drawdown (%)', 0):.2f}%",
            key=f"{filter_name}_metric_max_dd",
        )

    st.markdown("---")
    
    st.subheader("Rendimiento Detallado")
    tab1, tab2 = st.tabs(["Calendario", "Rendimiento Mensual"])

    with tab2:
        # --- INICIO DE LA SOLUCIÓN DEFINITIVA ---
        # 1. Agrupar y ordenar cronológicamente (sigue siendo necesario)
        monthly_pnl_chart = trades_df.set_index('entry_time_dt')['pnl_net'].resample('ME').sum().reset_index()
        monthly_pnl_chart.columns = ['Mes_dt', 'PnL']
        monthly_pnl_chart = monthly_pnl_chart.sort_values(by='Mes_dt', ascending=True)
        
        # 2. Crear las etiquetas de texto
        monthly_pnl_chart['Mes'] = monthly_pnl_chart['Mes_dt'].dt.strftime('%Y-%b')
        
        # 3. (CLAVE) Crear una lista de colores manual que coincida con el orden de las barras
        bar_colors = ['#26a69a' if pnl >= 0 else '#ef5350' for pnl in monthly_pnl_chart['PnL']]
        
        # 4. Usar plotly.graph_objects (go) para un control total
        fig_monthly_pnl = go.Figure()
        
        # 5. Añadir la traza de barras con los colores manuales
        fig_monthly_pnl.add_trace(go.Bar(
            x=monthly_pnl_chart['Mes'],
            y=monthly_pnl_chart['PnL'],
            marker_color=bar_colors,
            name='PnL Mensual'
        ))
        
        # 6. Configurar el layout para que coincida con el estilo anterior
        fig_monthly_pnl.update_layout(
            title_text='Rendimiento Mensual',
            xaxis_title='Mes',
            yaxis_title='Ganancia/Pérdida ($)',
            template="plotly_dark",
            showlegend=False # La leyenda de color ya no es necesaria
        )
        fig_monthly_pnl.update_xaxes(tickangle=45) 
        # --- FIN DE LA SOLUCIÓN DEFINITIVA ---
        
        st.plotly_chart(
            fig_monthly_pnl,
            use_container_width=True,
            key=f"{filter_name}_monthly_pnl",
        )

    with tab1:
        daily_pnl_agg = trades_df.set_index('entry_time_dt').groupby(pd.Grouper(freq='D'))['pnl_net'].agg(['sum', 'count'])
        daily_pnl_agg.columns = ['pnl', 'count']
        daily_pnl_agg = daily_pnl_agg[daily_pnl_agg['pnl'] != 0]
        daily_pnl_dict = {d.date(): {'pnl': r['pnl'], 'count': r['count']} for d, r in daily_pnl_agg.iterrows()}

        if daily_pnl_dict:
            start_date = st.session_state.ui_download_start
            
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            if nav_col1.button("◀ Mes Anterior", key=f"prev_month_{filter_name}"):
                st.session_state.calendar_month_offset -= 1
            if nav_col3.button("Mes Siguiente ▶", key=f"next_month_{filter_name}"):
                st.session_state.calendar_month_offset += 1
            
            current_month_start = (start_date.replace(day=1) + pd.DateOffset(months=st.session_state.calendar_month_offset))
            year, month = current_month_start.year, current_month_start.month
            
            monthly_pnl_value = daily_pnl_agg[
                (daily_pnl_agg.index.year == year) & (daily_pnl_agg.index.month == month)
            ]['pnl'].sum()
            
            month_start_timestamp = pd.Timestamp(year, month, 1, tz=st.session_state.ui_display_tz)
            
            equity_series = st.session_state.global_equity_history.copy()
            equity_series['time'] = pd.to_datetime(equity_series['time'], unit='s', utc=True).dt.tz_convert(st.session_state.ui_display_tz)
            equity_series.set_index('time', inplace=True)
            equity_series = equity_series['equity']
            
            previous_equity = equity_series[equity_series.index < month_start_timestamp]
            if not previous_equity.empty:
                monthly_start_equity = previous_equity.iloc[-1]
            else:
                monthly_start_equity = st.session_state.ui_initial_capital
            
            calendar_html = generate_calendar_html(daily_pnl_dict, year, month, monthly_pnl_value, monthly_start_equity)
            
            st.markdown(calendar_html, unsafe_allow_html=True)
        else:
            st.text("No hay datos para mostrar en el calendario.")

    st.markdown("---")
    st.subheader("Historial de Operaciones")
    trades_display = trades_df.copy()
    direction_map = {1.0: 'LONG', -1.0: 'SHORT'}
    exit_reason_map = {1.0: 'Take Profit', 2.0: 'Stop Loss', 3.0: 'Timeout'}
    
    trades_display['direction'] = trades_df_raw['direction'].map(direction_map).fillna('Desconocido')
    trades_display['exit_reason'] = trades_df_raw['exit_reason'].map(exit_reason_map).fillna('Desconocido')
    trades_display['entry_time_str'] = trades_display['entry_time_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    trades_display['exit_time_str'] = pd.to_datetime(trades_display['exit_time'], unit='s', utc=True).dt.tz_convert(st.session_state.ui_display_tz).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(
        trades_display[
            [
                'entry_time_str',
                'exit_time_str',
                'direction',
                'size',
                'entry_price',
                'exit_price',
                'pnl_net',
                'exit_reason',
            ]
        ],
        key=f"{filter_name}_trades_table",
    )

