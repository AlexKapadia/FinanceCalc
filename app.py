import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from arch import arch_model
from scipy.stats import t
import io, base64
import scipy.stats

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        # ------------------------------
        # PART 1: Input Validation
        # ------------------------------
        data = request.get_json()
        tickers = data.get('tickers')  # list of ticker strings
        amounts = data.get('amounts')  # list of investment amounts (numbers)

        if not tickers or not amounts or len(tickers) != len(amounts):
            return jsonify({'error': 'Invalid input: Please provide equal numbers of tickers and amounts.'}), 400

        # Validate tickers and amounts
        tickers = [ticker.strip().upper() for ticker in tickers]
        amounts = [float(amount) for amount in amounts]
        
        # Check for valid amounts
        if any(amount <= 0 for amount in amounts):
            return jsonify({'error': 'Investment amounts must be positive numbers.'}), 400

        # Normalize tickers and compute portfolio weights
        portfolio_size = sum(amounts)
        weights = np.array(amounts) / portfolio_size

        # Validate date range
        endDate = dt.datetime.now()
        startDate = endDate - dt.timedelta(days=500)
        
        # Ensure we have enough data points
        min_data_points = 100  # Minimum required for meaningful analysis
        
        # Download asset data with error handling
        try:
            if len(tickers) == 1:
                stockData = yf.download(tickers[0], start=startDate, end=endDate, progress=False, auto_adjust=True)
            else:
                stockData = yf.download(tickers, start=startDate, end=endDate, progress=False, auto_adjust=True)
        except Exception as e:
            return jsonify({'error': f'Error downloading data: {str(e)}'}), 500

        # Validate data completeness
        if len(stockData) < min_data_points:
            return jsonify({'error': f'Insufficient historical data. Need at least {min_data_points} data points.'}), 400

        # Check for missing data
        if stockData.isnull().any().any():
            return jsonify({'error': 'Missing data detected in the historical prices. Please try different tickers or time period.'}), 400

        # ------------------------------
        # PART 2: Portfolio Simulation & Stats
        # ------------------------------
        # Extract Price Data with validation
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
                            return jsonify({'error': "Could not extract price data. Please try different tickers."}), 400
        else:
            if 'Close' in stockData.columns:
                stock = stockData["Close"]
            elif 'Adj Close' in stockData.columns:
                stock = stockData["Adj Close"]
            else:
                return jsonify({'error': "Could not find price data. Please try different tickers."}), 400

        # Validate price data
        if (stock <= 0).any().any():
            return jsonify({'error': "Invalid price data detected. Please try different tickers."}), 400

        # Calculate returns with validation
        returns = stock.pct_change().dropna()
        if len(returns) < min_data_points:
            return jsonify({'error': 'Insufficient return data for analysis.'}), 400

        # Validate returns for extreme values
        if (returns.abs() > 0.5).any().any():  # More than 50% daily return is suspicious
            return jsonify({'error': 'Extreme returns detected. Data may be unreliable.'}), 400

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

        # Monte Carlo simulation parameters - Increased for better accuracy
        T = 100  # Increased back to 100 days for better long-term projection
        mc_sims = 1000  # Increased to 1000 simulations for better statistical significance
        min_iterations = 5  # Increased to 5 minimum iterations
        max_iterations = 20  # Increased to 20 maximum iterations
        target_std_dev = 0.005  # Decreased to 0.5% for tighter convergence
        df_t = 4  # Increased degrees of freedom for more realistic distribution
        initialportfolio = portfolio_size

        def simulate_jump_diffusion(initial_value, days, mean_return, volatility, df):
            # Generate random returns using t-distribution for more realistic fat tails
            Z = np.random.standard_t(df, size=days)
            
            # Add time-varying volatility (GARCH-like effect)
            vol_persistence = 0.85  # Volatility persistence parameter
            vol_shock = 0.15  # New shock weight
            time_varying_vol = np.zeros(days)
            time_varying_vol[0] = volatility
            
            for i in range(1, days):
                # Simple GARCH(1,1) approximation
                time_varying_vol[i] = (vol_persistence * time_varying_vol[i-1] + 
                                     vol_shock * abs(Z[i-1]) * volatility)
            
            # Generate jump process with time-varying intensity
            base_jump_prob = 0.02  # 2% base chance of a jump
            jump_intensity = np.maximum(0.01, base_jump_prob * (1 + 0.5 * abs(Z)))  # Higher jumps during volatile periods
            jumps = np.random.binomial(1, jump_intensity)
            
            # Jump magnitudes with asymmetric distribution (more negative jumps)
            jump_magnitudes = np.where(np.random.random(days) < 0.7,  # 70% chance of negative jump
                                     np.random.normal(-0.025, 0.04, days),  # Negative jumps (crashes)
                                     np.random.normal(0.015, 0.025, days))   # Positive jumps (rallies)
            
            # Combine components with time-varying volatility
            daily_returns = mean_return + time_varying_vol * Z + jumps * jump_magnitudes
            
            # Add realistic constraints with time-varying limits
            max_daily_change = 0.12 + 0.03 * abs(Z) / np.max(abs(Z))  # Dynamic limits based on market stress
            daily_returns = np.clip(daily_returns, -max_daily_change, max_daily_change)
            
            # Calculate portfolio values
            portfolio_values = initial_value * np.cumprod(1 + daily_returns)
            return np.insert(portfolio_values, 0, initial_value)  # Include initial value

        # Calculate portfolio weights
        weights = np.array(amounts) / portfolio_size

        # Calculate portfolio returns and volatility (scalar values)
        try:
            # Ensure returns data is properly aligned with tickers
            portfolio_returns = returns[tickers].dot(weights)
            portfolio_mean_return = float(portfolio_returns.mean())  # Ensure scalar
            portfolio_volatility_scalar = float(portfolio_returns.std())  # Ensure scalar
        except Exception as e:
            # Fallback calculation
            portfolio_returns = returns.mean(axis=1)  # Equal weighted if calculation fails
            portfolio_mean_return = float(portfolio_returns.mean())
            portfolio_volatility_scalar = float(portfolio_returns.std())

        # Initialize simulation parameters with enhanced tracking
        prev_metrics = {'std': float("inf"), 'mean': 0, 'skew': 0, 'kurt': 0}
        converged = False
        iteration = 0
        simulated_paths = []
        convergence_history = []

        while not converged and iteration < max_iterations:
            # Generate new paths using scalar portfolio metrics
            new_paths = np.array([simulate_jump_diffusion(initialportfolio, T, portfolio_mean_return, portfolio_volatility_scalar, df_t) 
                                for _ in range(mc_sims)])
            
            # Combine with previous paths
            simulated_paths = np.vstack([simulated_paths, new_paths]) if len(simulated_paths) > 0 else new_paths
            
            # Calculate multiple convergence metrics
            final_values = simulated_paths[:, -1]
            current_metrics = {
                'std': np.std(final_values),
                'mean': np.mean(final_values),
                'skew': scipy.stats.skew(final_values) if len(final_values) > 3 else 0,
                'kurt': scipy.stats.kurtosis(final_values) if len(final_values) > 4 else 0
            }
            
            # Multi-metric convergence checking
            std_change = abs(current_metrics['std'] - prev_metrics['std']) / prev_metrics['std'] if prev_metrics['std'] != float("inf") else float("inf")
            mean_change = abs(current_metrics['mean'] - prev_metrics['mean']) / abs(prev_metrics['mean']) if prev_metrics['mean'] != 0 else float("inf")
            
            # Track convergence history
            convergence_history.append({
                'iteration': iteration + 1,
                'paths': len(simulated_paths),
                'std_change': std_change,
                'mean_change': mean_change,
                'current_std': current_metrics['std'],
                'current_mean': current_metrics['mean']
            })
            
            # Enhanced convergence criteria
            if iteration >= min_iterations:
                # Primary convergence: standard deviation stability
                std_converged = std_change < target_std_dev
                # Secondary convergence: mean stability  
                mean_converged = mean_change < target_std_dev * 2 if mean_change != float("inf") else False
                # Tertiary convergence: sufficient sample size
                sample_converged = len(simulated_paths) >= 5000
                
                # Require at least 2 of 3 criteria for convergence
                convergence_score = sum([std_converged, mean_converged, sample_converged])
                if convergence_score >= 2:
                    converged = True
            
            prev_metrics = current_metrics.copy()
            iteration += 1
            
            # Adaptive simulation increase based on convergence rate
            if not converged and iteration < max_iterations:
                if std_change > target_std_dev * 5:  # Poor convergence
                    mc_sims += 1500  # Large increase
                elif std_change > target_std_dev * 2:  # Moderate convergence
                    mc_sims += 1000  # Medium increase
                else:  # Good convergence
                    mc_sims += 500   # Small increase

        # Validate final results with statistical tests
        final_values = simulated_paths[:, -1]
        
        # Test for normality (should NOT be normal due to fat tails and jumps)
        _, normality_p = scipy.stats.jarque_bera(final_values)
        
        # Calculate simulation quality metrics
        simulation_quality = {
            'total_paths': int(len(simulated_paths)),
            'iterations_used': int(iteration),
            'converged': bool(converged),
            'final_std_change': float(std_change) if std_change != float("inf") else None,
            'normality_rejected': bool(normality_p < 0.05),  # Good - means we have realistic fat tails
            'skewness': float(scipy.stats.skew(final_values)),
            'kurtosis': float(scipy.stats.kurtosis(final_values)),
            'convergence_history': [
                {
                    'iteration': int(h['iteration']),
                    'paths': int(h['paths']),
                    'std_change': float(h['std_change']) if h['std_change'] != float("inf") else None,
                    'mean_change': float(h['mean_change']) if h['mean_change'] != float("inf") else None,
                    'current_std': float(h['current_std']),
                    'current_mean': float(h['current_mean'])
                } for h in convergence_history[-5:]
            ]  # Last 5 iterations
        }

        # Calculate final statistics with enhanced precision
        mean_path = np.mean(simulated_paths, axis=0)
        confidence_intervals = {
            '95': np.percentile(simulated_paths, [2.5, 97.5], axis=0),
            '90': np.percentile(simulated_paths, [5, 95], axis=0),
            '80': np.percentile(simulated_paths, [10, 90], axis=0)
        }
        
        # Enhanced path statistics
        path_statistics = {
            'mean_final_value': float(np.mean(final_values)),
            'median_final_value': float(np.median(final_values)),
            'std_final_value': float(np.std(final_values)),
            'min_final_value': float(np.min(final_values)),
            'max_final_value': float(np.max(final_values)),
            'probability_of_loss': float(np.mean(final_values < initialportfolio)),
            'probability_of_doubling': float(np.mean(final_values > 2 * initialportfolio)),
            'expected_return': float((np.mean(final_values) - initialportfolio) / initialportfolio),
            'value_at_risk_5': float(np.percentile(final_values, 5)),
            'value_at_risk_1': float(np.percentile(final_values, 1))
        }

        # Generate enhanced plots with multiple confidence intervals
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        num_paths_to_plot = min(100, len(simulated_paths))  # Increased for better visualization
        
        # Plot individual paths with varying transparency
        sample_indices = np.random.choice(len(simulated_paths), num_paths_to_plot, replace=False)
        for i in sample_indices:
            ax1.plot(range(T+1), simulated_paths[i], color='lightgray', alpha=0.3, linewidth=0.5)
        
        # Plot mean path
        ax1.plot(range(T+1), mean_path, 'b-', linewidth=3, label='Mean Path')
        
        # Plot multiple confidence intervals
        ax1.fill_between(range(T+1), 
                        confidence_intervals['95'][0], 
                        confidence_intervals['95'][1], 
                        color='red', 
                        alpha=0.15, 
                        label='95% Confidence Interval')
        ax1.fill_between(range(T+1), 
                        confidence_intervals['90'][0], 
                        confidence_intervals['90'][1], 
                        color='orange', 
                        alpha=0.2, 
                        label='90% Confidence Interval')
        ax1.fill_between(range(T+1), 
                        confidence_intervals['80'][0], 
                        confidence_intervals['80'][1], 
                        color='green', 
                        alpha=0.25, 
                        label='80% Confidence Interval')
        
        # Add simulation quality annotations
        ax1.text(0.02, 0.98, f"Paths: {simulation_quality['total_paths']:,}\nIterations: {simulation_quality['iterations_used']}\nConverged: {simulation_quality['converged']}", 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
        
        ax1.set_title(f'Monte Carlo Portfolio Simulation\n({simulation_quality["total_paths"]:,} paths, {simulation_quality["iterations_used"]} iterations)')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Save plot as base64
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
        buf1.seek(0)
        img1 = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)

        # Enhanced second plot with statistics
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top subplot: Mean path and trend
        ax2.plot(range(T+1), mean_path, label="Mean Path", color="blue", linewidth=3)
        fit_coeffs = np.polyfit(np.arange(T+1), mean_path, deg=1)
        best_fit_line = np.polyval(fit_coeffs, np.arange(T+1))
        ax2.plot(range(T+1), best_fit_line, label="Trend Line", linestyle="--", color="red", linewidth=2)
        
        # Add percentile lines
        p75_line = np.percentile(simulated_paths, 75, axis=0)
        p25_line = np.percentile(simulated_paths, 25, axis=0)
        ax2.plot(range(T+1), p75_line, label="75th Percentile", color="green", alpha=0.7)
        ax2.plot(range(T+1), p25_line, label="25th Percentile", color="orange", alpha=0.7)
        
        ax2.set_ylabel("Portfolio Value ($)")
        ax2.set_title("Portfolio Simulation Paths with Trend Analysis")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Bottom subplot: Final value distribution
        ax3.hist(final_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        ax3.axvline(path_statistics['mean_final_value'], color='blue', linestyle='-', linewidth=2, label=f'Mean: ${path_statistics["mean_final_value"]:,.0f}')
        ax3.axvline(path_statistics['median_final_value'], color='green', linestyle='--', linewidth=2, label=f'Median: ${path_statistics["median_final_value"]:,.0f}')
        ax3.axvline(initialportfolio, color='red', linestyle=':', linewidth=2, label=f'Initial: ${initialportfolio:,.0f}')
        ax3.axvline(path_statistics['value_at_risk_5'], color='orange', linestyle='-.', linewidth=2, label=f'5% VaR: ${path_statistics["value_at_risk_5"]:,.0f}')
        
        ax3.set_xlabel("Final Portfolio Value ($)")
        ax3.set_ylabel("Density")
        ax3.set_title(f"Distribution of Final Values\nP(Loss): {path_statistics['probability_of_loss']:.1%}, P(Double): {path_statistics['probability_of_doubling']:.1%}")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
        buf2.seek(0)
        img2 = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)

        # Compute additional portfolio statistics using portfolio returns
        if len(tickers) > 1:
            cov_matrix = returns[tickers].cov()
            corr_matrix = returns[tickers].corr()
        else:
            var_val = returns[tickers[0]].var()
            cov_matrix = pd.DataFrame({tickers[0]: [var_val]}, index=[tickers[0]])
            corr_matrix = pd.DataFrame({tickers[0]: [1.0]}, index=[tickers[0]])
        
        portfolio_expected_daily_return = portfolio_mean_return
        portfolio_volatility_daily = portfolio_volatility_scalar
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

        # Add risk metrics using portfolio returns
        risk_free_rate = 0.05  # 5% annual risk-free rate
        daily_rf = risk_free_rate / 252
        
        # Calculate Sharpe Ratio with validation
        excess_returns = portfolio_expected_daily_return - daily_rf
        sharpe_ratio = np.sqrt(252) * excess_returns / portfolio_volatility_daily if portfolio_volatility_daily != 0 else 0
        
        # Calculate Sortino Ratio with validation - use portfolio returns
        negative_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.sqrt(np.mean(negative_returns**2)) if len(negative_returns) > 0 else portfolio_volatility_daily
        sortino_ratio = np.sqrt(252) * excess_returns / downside_deviation if downside_deviation != 0 else 0
        
        # Calculate Maximum Drawdown with validation - use portfolio returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        if isinstance(max_drawdown, pd.Series):
            max_drawdown = max_drawdown.iloc[0] if len(max_drawdown) > 0 else 0
        max_drawdown = float(max_drawdown)
        
        # Calculate Value at Risk (VaR) - use portfolio returns
        try:
            var_95 = float(np.percentile(portfolio_returns.dropna(), 5))
            var_99 = float(np.percentile(portfolio_returns.dropna(), 1))
        except:
            var_95 = float(portfolio_returns.quantile(0.05))
            var_99 = float(portfolio_returns.quantile(0.01))
        
        # Calculate Expected Shortfall (CVaR) - use portfolio returns
        try:
            cvar_95_series = portfolio_returns[portfolio_returns <= var_95]
            cvar_95 = float(cvar_95_series.mean()) if len(cvar_95_series) > 0 else var_95
            
            cvar_99_series = portfolio_returns[portfolio_returns <= var_99]
            cvar_99 = float(cvar_99_series.mean()) if len(cvar_99_series) > 0 else var_99
        except:
            cvar_95 = var_95
            cvar_99 = var_99
        
        # Calculate Information Ratio - use portfolio returns
        try:
            index_data = yf.download('^GSPC', start=startDate, end=endDate, progress=False, auto_adjust=True)
            if 'Close' in index_data.columns:
                index_prices = index_data['Close']
            elif 'Adj Close' in index_data.columns:
                index_prices = index_data['Adj Close']
            else:
                index_prices = index_data.iloc[:, 0]  # Use first column as fallback
            
            index_returns = index_prices.pct_change().dropna()
            
            # Align portfolio and index returns
            common_index = portfolio_returns.index.intersection(index_returns.index)
            if len(common_index) > 0:
                portfolio_aligned = portfolio_returns.loc[common_index]
                index_aligned = index_returns.loc[common_index]
                
                tracking_error = float((portfolio_aligned - index_aligned).std() * np.sqrt(252))
                portfolio_mean_annual = float(portfolio_aligned.mean() * 252)
                index_mean_annual = float(index_aligned.mean() * 252)
                information_ratio = (portfolio_mean_annual - index_mean_annual) / tracking_error if tracking_error != 0 else 0
            else:
                information_ratio = 0
        except:
            information_ratio = 0
        
        # Calculate Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Add to portfolio stats
        portfolio_stats.update({
            "sharpe_ratio": float(round(sharpe_ratio, 6)),
            "sortino_ratio": float(round(sortino_ratio, 6)),
            "max_drawdown": float(round(max_drawdown, 6)),
            "var_95": float(round(var_95, 6)),
            "var_99": float(round(var_99, 6)),
            "cvar_95": float(round(cvar_95, 6)),
            "cvar_99": float(round(cvar_99, 6)),
            "information_ratio": float(round(information_ratio, 6)),
            "calmar_ratio": float(round(calmar_ratio, 6)),
            "risk_free_rate": float(round(risk_free_rate, 6))
        })

        # Update financial terms dictionary with new metrics
        financial_terms = {
            "expected_daily_return": {
                "term": "Expected Daily Return",
                "definition": "The average return expected per day based on historical data. This is calculated by taking the mean of historical daily returns.",
                "formula": "Mean of daily returns = Σ(daily returns) / n",
                "interpretation": "A positive value indicates an expected gain, while a negative value suggests an expected loss. This is annualized by multiplying by 252 (trading days)."
            },
            "volatility_daily": {
                "term": "Daily Volatility",
                "definition": "A measure of the dispersion of returns around the mean, calculated as the standard deviation of daily returns.",
                "formula": "σ = √(Σ(returns - mean)² / n)",
                "interpretation": "Higher volatility indicates greater risk and price fluctuations. This is annualized by multiplying by √252."
            },
            "expected_annual_return": {
                "term": "Expected Annual Return",
                "definition": "The projected yearly return based on historical daily returns, annualized by multiplying by 252 trading days.",
                "formula": "Annual Return = Daily Return × 252",
                "interpretation": "This represents the expected yearly growth of your investment, assuming current market conditions persist."
            },
            "annual_volatility": {
                "term": "Annual Volatility",
                "definition": "The yearly measure of price fluctuations, calculated by annualizing daily volatility.",
                "formula": "Annual Volatility = Daily Volatility × √252",
                "interpretation": "A key risk metric - higher values indicate greater uncertainty in returns."
            },
            "diversification_ratio": {
                "term": "Diversification Ratio",
                "definition": "Measures the benefit of diversification in a portfolio. Values greater than 1 indicate effective diversification.",
                "formula": "DR = (Sum of weighted individual volatilities) / (Portfolio volatility)",
                "interpretation": "Higher values (above 1) indicate better diversification benefits. A value of 1 means no diversification benefit."
            },
            "sharpe_ratio": {
                "term": "Sharpe Ratio",
                "definition": "Measures risk-adjusted returns by comparing excess returns to volatility.",
                "formula": "Sharpe = (Portfolio Return - Risk-free Rate) / Portfolio Volatility",
                "interpretation": "Higher values indicate better risk-adjusted returns. Generally, >1 is good, >2 is very good, >3 is excellent."
            },
            "sortino_ratio": {
                "term": "Sortino Ratio",
                "definition": "Similar to Sharpe ratio but only considers downside volatility, focusing on negative returns.",
                "formula": "Sortino = (Portfolio Return - Risk-free Rate) / Downside Deviation",
                "interpretation": "Higher values indicate better risk-adjusted returns, specifically for downside risk protection."
            },
            "max_drawdown": {
                "term": "Maximum Drawdown",
                "definition": "The largest peak-to-trough decline in portfolio value over the specified period.",
                "formula": "Max Drawdown = (Peak Value - Trough Value) / Peak Value",
                "interpretation": "Lower values are better. This represents the worst-case scenario for portfolio decline."
            },
            "var_95": {
                "term": "95% Value at Risk (VaR)",
                "definition": "The maximum loss that is expected to be exceeded only 5% of the time.",
                "formula": "VaR(95%) = Percentile(returns, 5%)",
                "interpretation": "A VaR of -2% means there's a 95% chance the daily loss won't exceed 2%."
            },
            "cvar_95": {
                "term": "95% Conditional Value at Risk (CVaR)",
                "definition": "The average loss when the loss exceeds the 95% VaR.",
                "formula": "CVaR(95%) = Mean(returns | returns ≤ VaR(95%))",
                "interpretation": "The expected loss in the worst 5% of cases."
            },
            "information_ratio": {
                "term": "Information Ratio",
                "definition": "Measures the risk-adjusted return relative to a benchmark.",
                "formula": "IR = (Portfolio Return - Benchmark Return) / Tracking Error",
                "interpretation": "Higher values indicate better risk-adjusted performance relative to the benchmark."
            },
            "calmar_ratio": {
                "term": "Calmar Ratio",
                "definition": "Measures return relative to maximum drawdown.",
                "formula": "Calmar = Annual Return / |Maximum Drawdown|",
                "interpretation": "Higher values indicate better risk-adjusted returns relative to the worst drawdown."
            },
            "efficient_frontier": {
                "term": "Efficient Frontier",
                "definition": "A curve showing the set of optimal portfolios that offer the highest expected return for each level of risk (volatility). Each point on the frontier represents a portfolio that maximizes return for a given risk level.",
                "formula": "Optimized using: Maximize E(r) - λ/2 * σ², where E(r) is expected return, σ² is variance, and λ is risk aversion parameter",
                "interpretation": "The red star shows your current portfolio, green star shows the optimal portfolio. Portfolios on the curve are efficient - you can't get higher returns without taking more risk. The curve helps you see if your portfolio is optimally diversified."
            },
            "beta": {
                "term": "Beta",
                "definition": "Measures the portfolio's sensitivity to market movements (S&P 500).",
                "formula": "β = Covariance(Portfolio Returns, Market Returns) / Variance(Market Returns)",
                "interpretation": "β > 1: More volatile than market, β = 1: Same as market, β < 1: Less volatile than market"
            },
            "alpha": {
                "term": "Alpha",
                "definition": "Excess return of the portfolio compared to the market return, adjusted for risk (beta).",
                "formula": "α = Portfolio Return - [Risk-free Rate + β × (Market Return - Risk-free Rate)]",
                "interpretation": "Positive alpha indicates outperformance, negative alpha indicates underperformance."
            },
            "correlation_matrix": {
                "term": "Correlation Matrix",
                "definition": "Shows the relationship between different assets in the portfolio, ranging from -1 to 1.",
                "formula": "Correlation = Covariance(X,Y) / (σx × σy)",
                "interpretation": "1: Perfect positive correlation, 0: No correlation, -1: Perfect negative correlation. Lower correlations indicate better diversification benefits."
            },
            "covariance_matrix": {
                "term": "Covariance Matrix",
                "definition": "Shows how the returns of different assets move together. Higher values indicate assets move in the same direction more strongly.",
                "formula": "Covariance(X,Y) = E[(X - μx)(Y - μy)]",
                "interpretation": "Positive values mean assets tend to move together, negative values mean they move in opposite directions. Used to calculate portfolio risk through diversification."
            },
            "var_99": {
                "term": "99% Value at Risk (VaR)",
                "definition": "The maximum loss that is expected to be exceeded only 1% of the time.",
                "formula": "VaR(99%) = Percentile(returns, 1%)",
                "interpretation": "A VaR of -3% means there's a 99% chance the daily loss won't exceed 3%. More conservative than 95% VaR."
            },
            "cvar_99": {
                "term": "99% Conditional Value at Risk (CVaR)",
                "definition": "The average loss when the loss exceeds the 99% VaR.",
                "formula": "CVaR(99%) = Mean(returns | returns ≤ VaR(99%))",
                "interpretation": "The expected loss in the worst 1% of cases. Shows the severity of extreme losses."
            },
            "risk_free_rate": {
                "term": "Risk-Free Rate",
                "definition": "The theoretical rate of return of an investment with zero risk, typically based on government bonds.",
                "formula": "Usually 3-month Treasury bill rate or 10-year Treasury bond rate",
                "interpretation": "Used as a benchmark for calculating risk-adjusted returns like Sharpe ratio. Higher risk-free rates make risky investments relatively less attractive."
            },
            "optimal_portfolio": {
                "term": "Optimal Portfolio Weights",
                "definition": "The asset allocation that maximizes the Sharpe ratio (risk-adjusted return) for the given set of assets.",
                "formula": "Optimized using: w* = Σ⁻¹(μ - rf·1) / (1ᵀΣ⁻¹(μ - rf·1)), where Σ is covariance matrix, μ is expected returns",
                "interpretation": "Shows how you should allocate your money to achieve the best risk-adjusted returns. Compare with your current allocation to see potential improvements."
            }
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
        # PART 3: Portfolio Optimization
        # ------------------------------
        def optimize_portfolio(returns, risk_free_rate=0.05):
            n_assets = len(returns.columns)
            returns_mean = returns.mean()
            cov_matrix = returns.cov()
            
            # Generate random portfolios
            n_portfolios = 1000
            results = np.zeros((n_portfolios, n_assets + 2))
            
            for i in range(n_portfolios):
                weights = np.random.random(n_assets)
                weights = weights / np.sum(weights)
                
                portfolio_return = np.sum(returns_mean * weights) * 252
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                sharpe = (portfolio_return - risk_free_rate) / portfolio_std
                
                results[i, 0] = portfolio_std
                results[i, 1] = portfolio_return
                results[i, 2:] = weights
            
            # Find optimal portfolio (highest Sharpe ratio)
            optimal_idx = np.argmax(results[:, 1] / results[:, 0])
            optimal_weights = results[optimal_idx, 2:]
            
            return {
                "optimal_weights": dict(zip(returns.columns, optimal_weights)),
                "expected_return": float(results[optimal_idx, 1]),
                "expected_volatility": float(results[optimal_idx, 0])
            }
        
        # Calculate optimal portfolio
        if len(tickers) > 1:
            optimal_portfolio = optimize_portfolio(returns[tickers])
            portfolio_stats["optimal_portfolio"] = optimal_portfolio

        # ------------------------------
        # PART 4: Portfolio vs. Index Regression (Beta & Alpha)
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
        # Add Efficient Frontier plot
        if len(tickers) > 1:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            returns_mean = returns[tickers].mean() * 252
            cov_matrix = returns[tickers].cov() * 252
            
            # Generate random portfolios
            n_portfolios = 1000
            results = np.zeros((n_portfolios, 2))
            weights_list = []
            
            for i in range(n_portfolios):
                weights = np.random.random(len(tickers))
                weights = weights / np.sum(weights)
                weights_list.append(weights)
                
                portfolio_return = np.sum(returns_mean * weights)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                results[i, 0] = portfolio_std
                results[i, 1] = portfolio_return
            
            # Plot efficient frontier
            ax3.scatter(results[:, 0], results[:, 1], c=results[:, 1]/results[:, 0], 
                       marker='o', s=10, alpha=0.3, cmap='viridis')
            
            # Plot current portfolio
            current_std = portfolio_volatility_daily * np.sqrt(252)
            current_return = annual_return
            ax3.scatter(current_std, current_return, color='red', marker='*', 
                       s=200, label='Current Portfolio')
            
            # Plot optimal portfolio
            if 'optimal_portfolio' in portfolio_stats:
                opt_port = portfolio_stats['optimal_portfolio']
                ax3.scatter(opt_port['expected_volatility'], opt_port['expected_return'], 
                           color='green', marker='*', s=200, label='Optimal Portfolio')
            
            ax3.set_xlabel('Expected Volatility')
            ax3.set_ylabel('Expected Return')
            ax3.set_title('Efficient Frontier')
            ax3.legend()
            
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format='png')
            buf3.seek(0)
            img3 = base64.b64encode(buf3.read()).decode('utf-8')
            plt.close(fig3)
            
            # Add to response
            response = {
                'plot1': img1,
                'plot2': img2,
                'plot3': img3,
                'portfolio_stats': portfolio_stats,
                'index_stats': index_stats,
                'portfolio_comparison': portfolio_comparison,
                'financial_terms': financial_terms,
                'simulation_quality': simulation_quality,
                'path_statistics': path_statistics
            }
        else:
            response = {
                'plot1': img1,
                'plot2': img2,
                'portfolio_stats': portfolio_stats,
                'index_stats': index_stats,
                'portfolio_comparison': portfolio_comparison,
                'financial_terms': financial_terms,
                'simulation_quality': simulation_quality,
                'path_statistics': path_statistics
            }
        return jsonify(response)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
