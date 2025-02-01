import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from arch import arch_model
from scipy.stats import t
import io, base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        # ------------------------------
        # PART 1: Portfolio Simulation & Stats
        # ------------------------------
        data = request.get_json()
        tickers = data.get('tickers')  # list of ticker strings
        amounts = data.get('amounts')  # list of investment amounts (numbers)

        if not tickers or not amounts or len(tickers) != len(amounts):
            return jsonify({'error': 'Invalid input: Please provide equal numbers of tickers and amounts.'}), 400

        # Normalize tickers and compute portfolio weights
        tickers = [ticker.strip().upper() for ticker in tickers]
        portfolio_size = sum(amounts)
        weights = np.array(amounts) / portfolio_size

        endDate = dt.datetime.now()
        startDate = endDate - dt.timedelta(days=500)

        # Download asset data
        if len(tickers) == 1:
            stockData = yf.download(tickers[0], start=startDate, end=endDate, progress=False, auto_adjust=True)
        else:
            stockData = yf.download(tickers, start=startDate, end=endDate, progress=False, auto_adjust=True)

        # --- Extract Price Data ---
        if len(tickers) > 1 and isinstance(stockData.columns, pd.MultiIndex):
            try:
                stock = stockData.xs('Close', axis=1, level=1)
            except KeyError:
                try:
                    stock = stockData.xs('Adj Close', axis=1, level=1)
                except KeyError:
                    try:
                        stock = stockData.xs('Close', axis=1, level=0)
                    except KeyError:
                        try:
                            stock = stockData.xs('Adj Close', axis=1, level=0)
                        except KeyError:
                            raise KeyError("Could not extract 'Close' or 'Adj Close' from MultiIndex columns. Columns: " + str(stockData.columns))
        else:
            if 'Close' in stockData.columns:
                stock = stockData["Close"]
            elif 'Adj Close' in stockData.columns:
                stock = stockData["Adj Close"]
            else:
                raise KeyError("Neither 'Close' nor 'Adj Close' found in the columns.")

        if len(tickers) == 1:
            if isinstance(stock, pd.Series):
                stock = pd.DataFrame(stock, columns=[tickers[0]])
            stock.columns = [tickers[0]]

        returns = stock.pct_change()
        meanReturns = returns.mean()

        def estimate_volatility_garch(stock_returns):
            stock_returns = stock_returns.dropna() * 100
            try:
                model = arch_model(stock_returns, vol='Garch', p=1, q=1)
                garch_fit = model.fit(disp="off")
                return np.clip(garch_fit.conditional_volatility.iloc[-1], 0.0001, 0.05)
            except Exception:
                return np.std(stock_returns) / 100

        if len(tickers) == 1:
            mean_returns_array = np.array([meanReturns.iloc[0]])
            volatility_garch = np.array([estimate_volatility_garch(returns[tickers[0]])])
        else:
            volatility_garch_list = []
            for tkr in tickers:
                if tkr not in returns.columns:
                    raise ValueError(f"Ticker '{tkr}' not found in fetched data. Available columns: {returns.columns.tolist()}")
                vol = estimate_volatility_garch(returns[tkr])
                volatility_garch_list.append(vol)
            mean_returns_array = meanReturns[tickers].values.flatten()
            volatility_garch = np.array(volatility_garch_list)

        # Monte Carlo Simulation
        T = 100
        target_std_dev = 5
        prev_std_dev = float("inf")
        converged = False
        iteration = 0
        mc_sims = 1000
        min_iterations = 5
        df_t = 5
        initialportfolio = portfolio_size

        portfolio_sims = None
        while (not converged) or (iteration < min_iterations):
            iteration += 1
            portfolio_sims = np.zeros((T, mc_sims))
            def simulate_jump_diffusion(days):
                Z = t.rvs(df=df_t, size=(days, len(weights)))
                jumps = np.random.choice([0, 1], size=(days, len(weights)), p=[0.995, 0.005])
                jump_magnitude = np.random.normal(loc=-0.005, scale=0.02, size=(days, len(weights)))
                return mean_returns_array + volatility_garch * Z + jumps * jump_magnitude
            for m in range(mc_sims):
                simulated_returns = simulate_jump_diffusion(T)
                daily_return = np.dot(simulated_returns, weights)
                daily_return = np.clip(1 + daily_return, 0.95, 1.05)
                portfolio_values_sim = np.cumprod(daily_return) * initialportfolio
                portfolio_sims[:, m] = portfolio_values_sim
            mean_path = np.mean(portfolio_sims, axis=1)
            final_values = portfolio_sims[-1, :]
            current_std_dev = np.std(final_values)
            if abs(prev_std_dev - current_std_dev) < target_std_dev and iteration >= min_iterations:
                converged = True
            else:
                prev_std_dev = current_std_dev
                mc_sims += 100

        # Generate plots
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        num_paths_to_plot = min(500, mc_sims)
        ax1.plot(portfolio_sims[:, :num_paths_to_plot], color='gray', alpha=0.1)
        ax1.plot(mean_path, 'r', lw=2, label='Mean Path')
        ax1.set_title('Monte Carlo Simulation of Portfolio')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png')
        buf1.seek(0)
        img1 = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(mean_path, label="Mean Path", color="blue", linewidth=2)
        fit_coeffs = np.polyfit(np.arange(T), mean_path, deg=1)
        best_fit_line = np.polyval(fit_coeffs, np.arange(T))
        ax2.plot(best_fit_line, label="Best Fit Line", linestyle="--", color="red", linewidth=2)
        ax2.set_ylabel("Portfolio Value ($)")
        ax2.set_xlabel("Days")
        ax2.set_title("Monte Carlo Simulation - Mean Path & Best Fit Line")
        ax2.legend()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png')
        buf2.seek(0)
        img2 = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)

        # Compute additional portfolio statistics.
        if len(tickers) > 1:
            cov_matrix = returns[tickers].cov()
            corr_matrix = returns[tickers].corr()
        else:
            var_val = returns[tickers[0]].var()
            cov_matrix = pd.DataFrame({tickers[0]: [var_val]}, index=[tickers[0]])
            corr_matrix = pd.DataFrame({tickers[0]: [1.0]}, index=[tickers[0]])
        portfolio_expected_daily_return = np.dot(weights, mean_returns_array)
        if len(tickers) > 1:
            portfolio_volatility_daily = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
        else:
            portfolio_volatility_daily = np.sqrt(var_val)
        annual_return = portfolio_expected_daily_return * 252
        annual_volatility = portfolio_volatility_daily * np.sqrt(252)
        if len(tickers) > 1:
            indiv_vols = np.sqrt(np.diag(cov_matrix))
            diversification_ratio = np.sum(weights * indiv_vols) / portfolio_volatility_daily
        else:
            diversification_ratio = 1.0

        # Convert numpy types to native Python types
        portfolio_stats = {
            "expected_daily_return": float(round(portfolio_expected_daily_return, 6)),
            "volatility_daily": float(round(portfolio_volatility_daily, 6)),
            "expected_annual_return": float(round(annual_return, 6)),
            "annual_volatility": float(round(annual_volatility, 6)),
            "diversification_ratio": float(round(diversification_ratio, 6)),
            "covariance_matrix": cov_matrix.round(6).to_dict(),
            "correlation_matrix": corr_matrix.round(6).to_dict()
        }

        # ------------------------------
        # PART 2: Index (S&P 500) Comparison
        # ------------------------------
        indexData = yf.download('^GSPC', start=startDate, end=endDate, progress=False, auto_adjust=True)
        if 'Close' in indexData.columns:
            index_prices = indexData['Close']
        elif 'Adj Close' in indexData.columns:
            index_prices = indexData['Adj Close']
        else:
            raise KeyError("Could not extract 'Close' or 'Adj Close' for the index.")
        index_returns = index_prices.pct_change()
        index_mean_return = index_returns.mean()
        index_volatility_daily = index_returns.std()
        index_expected_annual_return = index_mean_return * 252
        index_annual_volatility = index_volatility_daily * np.sqrt(252)
        risk_free_rate_annual = 0.005
        risk_free_rate_daily = risk_free_rate_annual / 252
        index_sharpe = (index_expected_annual_return - risk_free_rate_annual) / index_annual_volatility
        index_stats = {
            "expected_daily_return": float(round(index_mean_return, 6)),
            "volatility_daily": float(round(index_volatility_daily, 6)),
            "expected_annual_return": float(round(index_expected_annual_return, 6)),
            "annual_volatility": float(round(index_annual_volatility, 6)),
            "sharpe_ratio": float(round(index_sharpe, 6))
        }

        # ------------------------------
        # PART 3: Portfolio vs. Index Regression (Beta & Alpha)
        # ------------------------------
        # Compute portfolio historical returns (weighted average)
        portfolio_hist_returns = returns.dot(weights)
        common_df = pd.concat([portfolio_hist_returns, index_returns], axis=1, join='inner').dropna()
        common_df.columns = ['portfolio', 'index']
        port_common = np.array(common_df['portfolio'].values).flatten()
        index_common = np.array(common_df['index'].values).flatten()
        # Ensure both arrays are 1D:
        port_common = port_common.flatten()
        index_common = index_common.flatten()
        beta, intercept = np.polyfit(index_common, port_common, 1)
        portfolio_alpha_daily = port_common.mean() - (risk_free_rate_daily + beta*(index_common.mean() - risk_free_rate_daily))
        portfolio_alpha_annual = portfolio_alpha_daily * 252
        portfolio_comparison = {
            "beta": float(round(beta, 6)),
            "alpha_daily": float(round(portfolio_alpha_daily, 6)),
            "alpha_annual": float(round(portfolio_alpha_annual, 6))
        }

        # ------------------------------
        # Merge all statistics into one response.
        # ------------------------------
        response = {
            'plot1': img1,
            'plot2': img2,
            'portfolio_stats': portfolio_stats,
            'index_stats': index_stats,
            'portfolio_comparison': portfolio_comparison
        }
        return jsonify(response)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
