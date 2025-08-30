# ui/comparison_renderer.py
"""
Módulo para renderizar el dashboard de comparación de resultados de backtests.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from agent_core.metrics import calculate_performance_metrics
from ui.results_renderer import render_global_results

def render_comparison_dashboard():
    """Muestra un dashboard comparando los resultados de múltiples filtros EMA."""
    st.header("Comparación de Filtros EMA")

    comparison_results = st.session_state.get('comparison_results', {})
    if not comparison_results:
        st.warning("No se encontraron resultados para la comparación.")
        return

    # --- 1. Tabla Resumen de Métricas ---
    st.subheader("Tabla Resumen de Métricas")
    
    metrics_data = []
    for filter_name, results in comparison_results.items():
        metrics = calculate_performance_metrics(
            results['trades'].to_dict('records'), 
            st.session_state.ui_initial_capital,
            results['equity'].values.tolist()
        )
        metrics_data.append({
            'Filtro': filter_name,
            'Ganancia Neta Total ($)': metrics.get('Ganancia Neta Total ($)', 0),
            'Ganancia Neta Total (%)': metrics.get('Ganancia Neta Total (%)', 0),
            'Win Rate (%)': metrics.get('Win Rate (%)', 0),
            'Profit Factor': metrics.get('Profit Factor', 'N/A'),
            'Max Drawdown (%)': metrics.get('Max Drawdown (%)', 0),
            'Trades Totales': metrics.get('Total Trades', 0)
        })

    df_metrics = pd.DataFrame(metrics_data).set_index('Filtro')
    st.dataframe(df_metrics.style.format({
        'Ganancia Neta Total ($)': "${:,.2f}",
        'Ganancia Neta Total (%)': "{:.2f}%",
        'Win Rate (%)': "{:.2f}%",
        'Max Drawdown (%)': "{:.2f}%"
    }))

    # --- 2. Gráfico de Curva de Equity Combinada ---
    st.subheader("Curvas de Equity Comparadas")
    fig = go.Figure()
    
    colors = {'Desactivado': '#5DADE2', 'Moderado': '#F5B041', 'Fuerte': '#58D68D'}

    for filter_name, results in comparison_results.items():
        equity_df = results['equity'].copy()
        equity_df['time'] = pd.to_datetime(equity_df['time'], unit='s', utc=True)
        equity_df = equity_df.set_index('time')
        
        # --- INICIO DE LA MODIFICACIÓN ---
        # Manejar duplicados y rellenar los días sin operaciones para una curva continua
        if equity_df.index.duplicated().any():
            equity_df = equity_df[~equity_df.index.duplicated(keep='last')]
        
        equity_df = equity_df.resample('D').last().ffill()
        # --- FIN DE LA MODIFICACIÓN ---
        
        fig.add_trace(go.Scatter(
            x=equity_df.index, 
            y=equity_df['equity'],
            mode='lines',
            name=filter_name,
            line=dict(color=colors.get(filter_name, 'white'))
        ))

    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Capital ($)",
        xaxis_title="Fecha",
        legend_title_text='Filtro EMA'
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 3. Vistas Detalladas en Pestañas ---
    st.subheader("Análisis Detallado por Filtro")
    
    tabs = st.tabs(comparison_results.keys())
    
    original_trades = st.session_state.session_trades
    original_equity = st.session_state.global_equity_history

    for i, filter_name in enumerate(comparison_results.keys()):
        with tabs[i]:
            st.session_state.session_trades = comparison_results[filter_name]['trades']
            st.session_state.global_equity_history = comparison_results[filter_name]['equity']
            
            render_global_results(filter_name=filter_name)

    st.session_state.session_trades = original_trades
    st.session_state.global_equity_history = original_equity