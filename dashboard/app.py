from select import select
import sys
import json
import os
from turtle import title
import streamlit as st
import pandas as pd

sys.path.append(f'{os.getcwd()}/src/')
sys.path.append(f'{os.getcwd()}/strategies/')
import config
from plot_controller import PlotController
from strategy_derivatives import StrategyDerivatives
from strategy_random import RandomStrategy
from create_folds import FoldController
from extract import ExtractData
from utils import prepare_asset_dataframe_format

class App:

    def __init__(self, tickers, interval) -> None:
        self._tickers = tickers
        self._interval = interval
        with open(f'{os.getcwd()}/dashboard/optimization_data/{self._interval}_optimization.json') as json_file:
            self._optimization_params = json.load(json_file)
        self._folds_dict = self._get_fold_dictionary()
        self._fig_dict = self._get_figure_dictionary() 
        self._random_fig_dicts = self._get_random_strategy_plots()

    def _get_fold_dictionary(self):        
        pass


    def _get_random_strategy_plots(self):
        data = prepare_asset_dataframe_format(self._tickers)
        fig_dict = RandomStrategy(data).run()
        return fig_dict

    def _get_figure_dictionary(self):
        fig_dict = {}
        data = prepare_asset_dataframe_format(self._tickers)
        
        fig_dict = PlotController(data=data, best_params=self._optimization_params['final_best_params']).run()
        return fig_dict


    def _plot_strategy_graph(self, strategy, selected_ticker):
        if strategy == 'Derivatives':
            return self._fig_dict[selected_ticker]
        elif strategy == 'Random':
            return self._random_fig_dicts[selected_ticker]

    def _populate_summary_tab(self, tab, selected_ticker, strategy_1, strategy_2):
        with tab:
            strategy1_column, strategy2_column = st.columns(2)
            with strategy1_column:
                fig1 = self._plot_strategy_graph(strategy_1, selected_ticker)
                st.markdown(f"<h1 style='text-align: center;'>{strategy_1}</h1>", unsafe_allow_html=True)
                st.plotly_chart(fig1)
            with strategy2_column:
                st.markdown(f"<h1 style='text-align: center;'>{strategy_2}</h1>", unsafe_allow_html=True)
                fig2 = self._plot_strategy_graph(strategy_2, selected_ticker)
                st.plotly_chart(fig2)

    def _plot_fold_data(self, pf, subplots, index):
        st.plotly_chart(pf.plot(subplots=subplots, title=f'{index} fold'))

    def _populate_fold_tab(self, tab, selected_ticker, subplots):

        data = prepare_asset_dataframe_format([selected_ticker])

        _, _, _, oos_dates = FoldController(data, fold_type='walk-forward').run()
        with tab:
            general_params, own_params = st.columns(2) 
            
            with general_params:
                st.markdown(f"<h1 style='text-align: center;'>Best params of all folds</h1>", unsafe_allow_html=True)
            
            with own_params:
                st.markdown(f"<h1 style='text-align: center;'>Best params of each fold</h1>", unsafe_allow_html=True)

            fold_final_params = self._optimization_params['final_best_params']
            
            for index in range(len(oos_dates)):
                
                fold_params = self._optimization_params[str(index)]['best_params']

                fold_data = data.loc[oos_dates[index]].copy()
                pf_best = StrategyDerivatives(fold_data.iloc[:,0], **fold_final_params).run()
                pf_own = StrategyDerivatives(fold_data.iloc[:,0], **fold_params).run()
                
                with general_params:
                    self._plot_fold_data(pf_best, subplots, index)

                with own_params:
                    self._plot_fold_data(pf_own, subplots, index)

    def _populate_subplots(self, cum_returns, trades,  drawdowns):
        subplots = []
        if cum_returns:
            subplots.append('cum_returns')
        if trades:
            subplots.append('trades')
        if drawdowns:
            subplots.append('drawdowns')
        return subplots

    def run(self):
        # Set title
        st.set_page_config(layout="wide")

        with st.sidebar:
            selected_ticker = st.selectbox('Ticker', self._tickers)
            strategy_1 = st.selectbox('Strategy 1', config.STRATEGY_LIST)
            strategy_2 = st.selectbox('Strategy 2', config.STRATEGY_LIST, index=1)
            st.text('Choose graph types for Fold tab:')
            cum_returns = st.checkbox('cum_returns', value=True)
            trades = st.checkbox('trades')
            drawdowns = st.checkbox('drawdowns')


        # Set tabs
        t1, t2, t3 = st.tabs(['Best Parameters', 'Fold Analysis', 'Others'])
        subplots = self._populate_subplots(cum_returns, trades, drawdowns)

        self._populate_summary_tab(t1, selected_ticker, strategy_1, strategy_2)
        self._populate_fold_tab(t2, selected_ticker, subplots)


if __name__ == '__main__':
    tickers = pd.read_csv(f'{os.getcwd()}/data/all_tickers.csv').columns.values.tolist()
    tickers.remove('timestamp')
    app = App(tickers=tickers, interval='1d')
    app.run()
    # data = prepare_asset_dataframe_format(tickers)


