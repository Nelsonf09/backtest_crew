# ui/comparison_renderer.py
"""
Módulo para renderizar el dashboard de comparación de resultados de backtests.
"""
import logging

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from agent_core.utils.metrics import compute_global_metrics
from shared.metrics import drawdown_stats
from ui.results_renderer import render_global_results

def render_comparison_dashboard():
    """Muestra un dashboard comparando los resultados de múltiples ejecuciones."""
    # El título ahora es dinámico basado en el tipo de comparación que se ejecutó.
    comparison_type = st.session_state.get('comparison_type', 'Resultados')
    st.header(f"Comparación de {comparison_type}")

    comparison_results = st.session_state.get('comparison_results', {})
    if not comparison_results:
        st.warning("No se encontraron resultados para la comparación.")
        return

    # --- 1. Tabla Resumen de Métricas ---
    st.subheader("Tabla Resumen de Métricas")
    
    metrics_data = []
    dd_curves = {}
    # La clave del diccionario (ej. 'Fuerte' o '5 mins') se usa como nombre.
    for name, results in comparison_results.items():
        equity_df = results['equity'].copy()
        equity_df['time'] = pd.to_datetime(equity_df['time'], unit='s', utc=True)
        eq_series = equity_df.set_index('time')['equity']

        trades_df = results['trades']
        trades_list = trades_df.rename(columns={'pnl_net': 'pnl'}).to_dict('records')
        metrics = compute_global_metrics(
            eq_series,
            trades_list,
            st.session_state.ui_initial_capital,
        )
        results['metrics'] = metrics

        dd_curve_pct = drawdown_stats(eq_series)["dd_series_pct"]
        dd_curves[name] = dd_curve_pct

        dd_from_curve = float(-dd_curve_pct.min()) if len(dd_curve_pct) else 0.0
        if abs(round(dd_from_curve, 2) - round(metrics['Max Drawdown (%)'], 2)) > 0.05:
            logging.warning(
                "Max DD mismatch for %s: metrics=%.2f%% curve=%.2f%%",
                name,
                metrics['Max Drawdown (%)'],
                dd_from_curve,
            )

        metrics_data.append({
            'Configuración': name,
            'Ganancia Neta Total ($)': metrics.get('Ganancia Neta Total ($)', 0),
            'Ganancia Neta Total (%)': metrics.get('Ganancia Neta Total (%)', 0),
            'Win Rate (%)': metrics.get('Win Rate (%)', 0),
            'Profit Factor': metrics.get('Profit Factor', 0),
            'Max Drawdown (%)': metrics.get('Max Drawdown (%)', 0),
            'Trades Totales': metrics.get('Trades Totales', 0)
        })

    df_metrics = pd.DataFrame(metrics_data).set_index('Configuración')
    st.session_state.metrics = df_metrics
    # Mostrar las métricas en formato porcentual positivo
    st.dataframe(df_metrics.style.format({
        'Ganancia Neta Total ($)': "${:,.2f}",
        'Ganancia Neta Total (%)': "{:.2f}%",
        'Win Rate (%)': "{:.2f}%",
        'Max Drawdown (%)': "{:.2f}%"
    }))

    # --- 2. Gráfico de Curva de Equity Combinada ---
    st.subheader("Curvas de Equity Comparadas")
    fig = go.Figure()
    
    # Paleta de colores genérica para que funcione con cualquier número de comparaciones
    colors = ['#5DADE2', '#F5B041', '#58D68D', '#EC7063', '#AF7AC5', '#48C9B0']

    # Se itera sobre los items para tener acceso al nombre y al color.
    for i, (name, results) in enumerate(comparison_results.items()):
        equity_df = results['equity'].copy()
        equity_df['time'] = pd.to_datetime(equity_df['time'], unit='s', utc=True)
        equity_df = equity_df.set_index('time')
        
        if equity_df.index.duplicated().any():
            equity_df = equity_df[~equity_df.index.duplicated(keep='last')]
        
        equity_df = equity_df.resample('D').last().ffill()
        
        fig.add_trace(go.Scatter(
            x=equity_df.index, 
            y=equity_df['equity'],
            mode='lines',
            name=name, # El nombre en la leyenda es la clave del diccionario
            line=dict(color=colors[i % len(colors)]) # Asigna colores cíclicamente
        ))

    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Capital ($)",
        xaxis_title="Fecha",
        legend_title_text=comparison_type
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Curvas de Drawdown Comparadas")
    if dd_curves:
        dd_df = pd.concat(dd_curves.values(), axis=1, keys=dd_curves.keys())
        dd_df.sort_index(inplace=True)
        dd_df = dd_df.ffill()
        st.session_state.drawdown_series = dd_df
        fig_ddc = go.Figure()
        for name in dd_df.columns:
            fig_ddc.add_trace(
                go.Scatter(
                    x=dd_df.index,
                    y=dd_df[name],
                    mode="lines",
                    name=name,
                    connectgaps=False,
                )
            )
        fig_ddc.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
        )
        st.plotly_chart(
            fig_ddc, use_container_width=True
        )

    # --- 3. Vistas Detalladas en Pestañas ---
    st.subheader("Análisis Detallado por Configuración")
    
    tabs = st.tabs(comparison_results.keys())
    
    # Guardar estado original para restaurarlo después
    original_trades = st.session_state.get('session_trades')
    original_equity = st.session_state.get('global_equity_history')

    for i, name in enumerate(comparison_results.keys()):
        with tabs[i]:
            # Establecer temporalmente los datos para la pestaña actual
            st.session_state.session_trades = comparison_results[name]['trades']
            st.session_state.global_equity_history = comparison_results[name]['equity']
            
            # Pasamos un nombre de filtro único para las keys de los widgets internos
            render_global_results(filter_name=f"comp_{name.replace(' ', '_')}")

    # Restaurar el estado original
    st.session_state.session_trades = original_trades
    st.session_state.global_equity_history = original_equity
