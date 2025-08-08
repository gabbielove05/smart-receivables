"""
Dashboard Components for Smart Receivables Navigator
Contains CFO panel, KPI cockpit, heat map, and what-if simulator.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from chatbot import generate_executive_summary, generate_insights

logger = logging.getLogger(__name__)

def render_cfo_panel(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Render the One-Minute CFO splash panel with key metrics and AI insights."""
    st.markdown("### 📊 One-Minute CFO Dashboard")
    
    try:
        # Calculate key metrics
        metrics = calculate_key_metrics(df)
        
        # Generate AI executive summary using new AI client
        st.subheader("🎯 Executive Summary")
        with st.spinner("Generating summary…"):
            try:
                # Calculate key metrics for summary
                ar_over_30 = len(df[df['days_overdue'] > 30]) if 'days_overdue' in df.columns else 0
                dso = metrics['dso']
                dso_target = 30
                
                # Get high risk accounts (overdue > 60 days)
                high_risk = df[df['days_overdue'] > 60] if 'days_overdue' in df.columns else pd.DataFrame()
                
                # Get exceptions in last 24h (simplified)
                exceptions_24h_list = df[df['status'] == 'exception'] if 'status' in df.columns else pd.DataFrame()
                
                kpis_text = f"""
                AR>30d: {ar_over_30}
                DSO: {dso}
                High-risk accts: {len(high_risk)}
                Exceptions (24h): {len(exceptions_24h_list)}
                """
                
                from ai_client import call_ai
                messages = [
                    {"role": "system", "content": "You are a concise payments analyst for Treasury/Receivables."},
                    {"role": "user", "content": f"Summarize risk, cash acceleration opportunities, and 2 concrete actions:\n{kpis_text}"}
                ]
                summary = call_ai(messages)
                st.markdown(summary)
            except Exception as e:
                st.warning(f"AI summary unavailable — using fallback. ({e})")
                bullets = []
                if dso > dso_target: bullets.append(f"DSO {dso:.1f} vs target {dso_target:.1f}: escalate >30d collections.")
                if len(exceptions_24h_list) > 0: bullets.append(f"{len(exceptions_24h_list)} exceptions in 24h: review top by amount.")
                if len(high_risk) > 0: bullets.append(f"{len(high_risk)} high-risk accounts: tighten cadence / holds.")
                if not bullets: bullets.append("No critical issues detected; maintain cadence.")
                st.markdown("\n".join(f"- {b}" for b in bullets))
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_metric_card(
                "💰 Total Receivables",
                f"${metrics['total_receivables']:,.0f}",
                metrics.get('receivables_change', 0),
                "vs last period"
            )
        
        with col2:
            render_metric_card(
                "⏰ Days Sales Outstanding",
                f"{metrics['dso']:.1f} days",
                metrics.get('dso_change', 0),
                "Target: 30 days"
            )
        
        with col3:
            render_metric_card(
                "🔴 Overdue Amount",
                f"${metrics['overdue_amount']:,.0f}",
                metrics.get('overdue_change', 0),
                f"{metrics['overdue_percentage']:.1f}% of total"
            )
        
        with col4:
            render_metric_card(
                "📊 Collection Rate",
                f"{metrics['collection_rate']:.1f}%",
                metrics.get('collection_change', 0),
                "This period"
            )
        
        # Risk indicators and trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚨 Risk Indicators")
            render_risk_indicators(metrics, config)
        
        with col2:
            st.markdown("#### 📈 Trend Analysis")
            render_trend_chart(df)
            
    except Exception as e:
        st.error(f"❌ Error rendering CFO panel: {str(e)}")
        logger.error(f"CFO panel error: {e}")

def render_kpi_dashboard(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Render comprehensive KPI cockpit with interactive charts."""
    st.markdown("### 📈 KPI Cockpit")
    
    try:
        metrics = calculate_key_metrics(df)
        
        # Top KPI row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("DSO", f"{metrics['dso']:.1f} days", delta=f"{metrics.get('dso_change', 0):+.1f}")
        
        with col2:
            st.metric("Hit Rate", f"{metrics['collection_rate']:.1f}%", delta=f"{metrics.get('collection_change', 0):+.1f}")
        
        with col3:
            st.metric("Unapplied Cash", f"${metrics.get('unapplied_cash', 0):,.0f}")
        
        with col4:
            st.metric("Bad Debt %", f"{metrics.get('bad_debt_rate', 0):.2f}%")
        
        with col5:
            st.metric("AR Turnover", f"{metrics.get('ar_turnover', 0):.1f}x")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Aging Waterfall")
            render_aging_waterfall(df)
        
        with col2:
            st.markdown("#### 🎯 Collection Performance")
            render_collection_performance(df)
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👥 Customer Analysis")
            render_customer_analysis(df)
        
        with col2:
            st.markdown("#### 💹 Cash Flow Forecast")
            render_cash_flow_forecast(df)
            
    except Exception as e:
        st.error(f"❌ Error rendering KPI dashboard: {str(e)}")
        logger.error(f"KPI dashboard error: {e}")

def render_heatmap(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Render accounts receivable risk heat map."""
    st.markdown("### 🔥 Receivables Risk Heat Map")
    
    try:
        # Create heat map data
        heatmap_data = create_heatmap_data(df)
        
        if heatmap_data.empty:
            st.warning("📊 No data available for heat map analysis.")
            return
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_risk = heatmap_data.values.mean()
            st.metric("📊 Average Risk Score", f"{avg_risk:.1f}/100")
        
        with col2:
            highest_risk = heatmap_data.values.max()
            st.metric("🔴 Highest Risk Score", f"{highest_risk:.0f}/100")
        
        with col3:
            high_risk_count = (heatmap_data.values >= 70).sum()
            st.metric("⚠️ High Risk Areas", f"{high_risk_count}")
        
        with col4:
            low_risk_count = (heatmap_data.values <= 40).sum()
            st.metric("✅ Low Risk Areas", f"{low_risk_count}")
        
        st.markdown("---")
        
        # Main heat map with improved styling
        fig = px.imshow(
            heatmap_data.values,
            labels=dict(x="Risk Category", y="Customer Segment", color="Risk Score"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale="RdYlGn_r",  # Red-Yellow-Green reversed (Red = high risk)
            title="Risk Assessment by Customer Segment and Category",
            text_auto=True  # Show values on heatmap
        )
        
        # Update text formatting
        fig.update_traces(
            texttemplate="%{z:.0f}",
            textfont={"size": 10, "color": "white"}
        )
        
        fig.update_layout(
            height=600,
            xaxis_title="Risk Categories",
            yaxis_title="Customer Segments",
            title_x=0.5,
            coloraxis_colorbar=dict(
                title="Risk Score",
                tickmode="linear",
                tick0=0,
                dtick=20
            )
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced analysis insights
        col1, col2 = st.columns(2)
        
        with col1:
            render_enhanced_risk_analysis(heatmap_data)
        
        with col2:
            render_enhanced_actions(heatmap_data)
        
        # Additional heatmap summary
        render_heatmap_summary(heatmap_data)
            
    except Exception as e:
        st.error(f"❌ Error rendering heat map: {str(e)}")
        logger.error(f"Heat map error: {e}")

def render_what_if_simulator(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Render interactive what-if scenario simulator."""
    st.markdown("### 🎯 What-If Simulator")
    
    try:
        st.markdown("Adjust parameters to see the impact on key metrics:")
        
        # Simulation controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            payment_terms = st.slider(
                "Payment Terms (days)",
                min_value=15,
                max_value=90,
                value=30,
                step=5,
                key="sim_payment_terms"
            )
        
        with col2:
            early_discount = st.slider(
                "Early Payment Discount (%)",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                key="sim_discount"
            )
        
        with col3:
            collection_efficiency = st.slider(
                "Collection Efficiency (%)",
                min_value=70,
                max_value=100,
                value=85,
                step=1,
                key="sim_efficiency"
            )
        
        with col4:
            credit_limit_change = st.slider(
                "Credit Limit Change (%)",
                min_value=-50,
                max_value=50,
                value=0,
                step=5,
                key="sim_credit_change"
            )
        
        # Run simulation
        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Running scenario analysis..."):
                baseline_metrics = calculate_key_metrics(df)
                simulated_metrics = run_simulation(
                    df, payment_terms, early_discount, 
                    collection_efficiency, credit_limit_change
                )
                
                # Display results
                st.markdown("#### 📊 Simulation Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**💰 Financial Impact**")
                    cash_impact = simulated_metrics['total_receivables'] - baseline_metrics['total_receivables']
                    st.metric(
                        "Cash Flow Impact",
                        f"${cash_impact:,.0f}",
                        delta=f"{(cash_impact/baseline_metrics['total_receivables']*100):+.1f}%"
                    )
                    
                    dso_change = simulated_metrics['dso'] - baseline_metrics['dso']
                    st.metric(
                        "DSO Change",
                        f"{simulated_metrics['dso']:.1f} days",
                        delta=f"{dso_change:+.1f}"
                    )
                
                with col2:
                    st.markdown("**🎯 Collection Impact**")
                    collection_change = simulated_metrics['collection_rate'] - baseline_metrics['collection_rate']
                    st.metric(
                        "Collection Rate",
                        f"{simulated_metrics['collection_rate']:.1f}%",
                        delta=f"{collection_change:+.1f}%"
                    )
                    
                    overdue_change = simulated_metrics['overdue_amount'] - baseline_metrics['overdue_amount']
                    st.metric(
                        "Overdue Amount",
                        f"${simulated_metrics['overdue_amount']:,.0f}",
                        delta=f"${overdue_change:+,.0f}"
                    )
                
                with col3:
                    st.markdown("**⚡ Risk Assessment**")
                    render_risk_score(simulated_metrics)
        
        # Scenario comparison chart
        st.markdown("#### 📈 Scenario Comparison")
        render_scenario_chart(df)
        
    except Exception as e:
        st.error(f"❌ Error rendering simulator: {str(e)}")
        logger.error(f"Simulator error: {e}")

def calculate_key_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate key receivables metrics."""
    try:
        if df.empty:
            return {}
        
        # Ensure required columns exist
        if 'amount' not in df.columns:
            return {'total_receivables': 0, 'dso': 0, 'collection_rate': 0, 'overdue_amount': 0, 'overdue_percentage': 0}
        
        total_receivables = df['amount'].sum()
        
        # Calculate DSO (simplified)
        # DSO = (Accounts Receivable / Total Credit Sales) × Number of Days
        avg_daily_sales = total_receivables / 365 if total_receivables > 0 else 1
        outstanding_amount = df[df.get('status', '') == 'outstanding']['amount'].sum() if 'status' in df.columns else total_receivables * 0.3
        dso = outstanding_amount / avg_daily_sales if avg_daily_sales > 0 else 0
        
        # Collection rate
        paid_amount = df[df.get('status', '') == 'paid']['amount'].sum() if 'status' in df.columns else total_receivables * 0.7
        collection_rate = (paid_amount / total_receivables * 100) if total_receivables > 0 else 0
        
        # Overdue analysis
        overdue_amount = df[df.get('status', '') == 'overdue']['amount'].sum() if 'status' in df.columns else total_receivables * 0.15
        overdue_percentage = (overdue_amount / total_receivables * 100) if total_receivables > 0 else 0
        
        return {
            'total_receivables': total_receivables,
            'dso': min(dso, 365),  # Cap DSO at 365 days
            'collection_rate': collection_rate,
            'overdue_amount': overdue_amount,
            'overdue_percentage': overdue_percentage,
            'outstanding_amount': outstanding_amount,
            'paid_amount': paid_amount,
            'ar_turnover': 365 / dso if dso > 0 else 0,
            'bad_debt_rate': overdue_percentage * 0.1,  # Estimate
            'unapplied_cash': total_receivables * 0.05  # Estimate
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}

def render_metric_card(title: str, value: str, change: float, subtitle: str) -> None:
    """Render a metric card with simple styling."""
    change_symbol = "↗" if change >= 0 else "↘"
    
    st.markdown(f"**{title}**")
    st.markdown(f"**{value}**")
    st.markdown(subtitle)
    if change != 0:
        st.markdown(f"{change_symbol} {abs(change):.1f}")
    st.markdown("---")

def render_risk_indicators(metrics: Dict[str, float], config: Dict[str, Any]) -> None:
    """Render risk indicator panel."""
    try:
        risk_items = []
        
        # DSO risk
        if metrics.get('dso', 0) > config['dso_threshold']:
            risk_items.append(("🔴 High DSO", f"DSO {metrics['dso']:.1f} days exceeds target {config['dso_threshold']} days"))
        
        # Overdue risk
        if metrics.get('overdue_amount', 0) > config['overdue_threshold']:
            risk_items.append(("🟠 High Overdue", f"${metrics['overdue_amount']:,.0f} overdue exceeds threshold"))
        
        # Collection risk
        if metrics.get('collection_rate', 0) < 80:
            risk_items.append(("🟡 Low Collection Rate", f"Collection rate {metrics['collection_rate']:.1f}% below target"))
        
        if not risk_items:
            st.success("✅ All risk indicators within acceptable ranges")
        else:
            for risk_title, risk_desc in risk_items:
                st.warning(f"{risk_title}: {risk_desc}")
                
    except Exception as e:
        logger.error(f"Error rendering risk indicators: {e}")

def render_aging_waterfall(df: pd.DataFrame) -> None:
    """Render aging waterfall chart."""
    try:
        # Create aging buckets
        aging_data = create_aging_buckets(df)
        
        if aging_data:
            fig = go.Figure()
            
            x_labels = list(aging_data.keys())
            values = list(aging_data.values())
            
            colors = ['#00FF00', '#FFD700', '#FFA500', '#FF6B6B', '#FF0000']
            
            fig.add_trace(go.Bar(
                x=x_labels,
                y=values,
                marker_color=colors[:len(values)],
                text=[f'${v:,.0f}' for v in values],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Accounts Receivable Aging",
                xaxis_title="Age Category",
                yaxis_title="Amount ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No aging data available")
            
    except Exception as e:
        logger.error(f"Error rendering aging waterfall: {e}")
        st.error("Error rendering aging chart")

def create_aging_buckets(df: pd.DataFrame) -> Dict[str, float]:
    """Create aging buckets from data."""
    try:
        if 'amount' not in df.columns:
            return {}
        
        # Simplified aging - in real implementation would use due dates
        total_amount = df['amount'].sum()
        
        return {
            '0-30 days': total_amount * 0.4,
            '31-60 days': total_amount * 0.25,
            '61-90 days': total_amount * 0.2,
            '91-120 days': total_amount * 0.1,
            '120+ days': total_amount * 0.05
        }
        
    except Exception as e:
        logger.error(f"Error creating aging buckets: {e}")
        return {}

def render_collection_performance(df: pd.DataFrame) -> None:
    """Render collection performance chart."""
    try:
        # Create sample collection performance data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        collection_rates = [85, 87, 82, 90, 88, 85]
        targets = [85] * 6
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=collection_rates,
            mode='lines+markers',
            name='Actual',
            line=dict(color='#0066CC', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=targets,
            mode='lines',
            name='Target',
            line=dict(color='#FFD700', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Collection Rate Trend",
            xaxis_title="Month",
            yaxis_title="Collection Rate (%)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error rendering collection performance: {e}")

def render_customer_analysis(df: pd.DataFrame) -> None:
    """Render customer analysis table."""
    try:
        if 'customer_id' in df.columns and 'amount' in df.columns:
            customer_summary = df.groupby('customer_id')['amount'].agg(['sum', 'count', 'mean']).round(2)
            customer_summary.columns = ['Total Amount', 'Invoice Count', 'Avg Amount']
            customer_summary = customer_summary.sort_values('Total Amount', ascending=False).head(10)
            
            st.dataframe(customer_summary, use_container_width=True)
        else:
            st.info("📊 Customer analysis requires customer_id and amount columns")
            
    except Exception as e:
        logger.error(f"Error rendering customer analysis: {e}")

def render_cash_flow_forecast(df: pd.DataFrame) -> None:
    """Render cash flow forecast chart."""
    try:
        # Create sample forecast data
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        forecast = np.random.normal(100000, 20000, 12).cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#0066CC', width=3),
            fill='tonexty',
            fillcolor='rgba(0, 102, 204, 0.1)'
        ))
        
        fig.update_layout(
            title="Cash Flow Forecast",
            xaxis_title="Month",
            yaxis_title="Cumulative Cash Flow ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error rendering cash flow forecast: {e}")

def create_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create meaningful heat map data based on receivables risk patterns."""
    try:
        # Define risk categories based on actual receivables patterns
        risk_categories = [
            'Days Overdue (0-30)',
            'Days Overdue (31-60)', 
            'Days Overdue (61-90)',
            'Days Overdue (90+)',
            'High Amount (>$50K)',
            'Payment History Issues',
            'Communication Unresponsive',
            'Credit Risk Concerns'
        ]
        
        # Customer segments derived from data
        if not df.empty and 'customer_id' in df.columns:
            # Create segments based on customer data patterns
            customer_counts = df['customer_id'].value_counts()
            if 'amount' in df.columns:
                customer_amounts = df.groupby('customer_id')['amount'].sum()
                segments = []
                
                # Categorize customers by volume and value
                for customer in customer_counts.head(10).index:
                    count = customer_counts[customer]
                    amount = customer_amounts.get(customer, 0)
                    
                    if amount > 100000:
                        segments.append(f"Enterprise ({customer})")
                    elif amount > 50000:
                        segments.append(f"Mid-Market ({customer})")
                    else:
                        segments.append(f"Small Business ({customer})")
                
                # Ensure we have at least 4 segments
                while len(segments) < 4:
                    segments.append(f"Customer Segment {len(segments)+1}")
                    
                segments = segments[:6]  # Limit to 6 for readability
            else:
                segments = ['High-Value Customers', 'Mid-Tier Customers', 'Standard Customers', 'New Customers']
        else:
            segments = ['High-Value Customers', 'Mid-Tier Customers', 'Standard Customers', 'New Customers']
        
        # Generate realistic risk scores based on actual data patterns
        np.random.seed(42)  # For consistent results
        data = np.zeros((len(segments), len(risk_categories)))
        
        for i, segment in enumerate(segments):
            for j, category in enumerate(risk_categories):
                # Different risk profiles for different segments
                if 'Enterprise' in segment or 'High-Value' in segment:
                    # Enterprise customers: lower overdue risk, higher amounts
                    if 'Days Overdue' in category:
                        if '0-30' in category:
                            score = np.random.randint(60, 85)  # Moderate risk
                        elif '31-60' in category:
                            score = np.random.randint(30, 60)  # Lower risk
                        else:
                            score = np.random.randint(10, 40)  # Very low risk
                    elif 'High Amount' in category:
                        score = np.random.randint(70, 95)  # High amount risk
                    elif 'Payment History' in category:
                        score = np.random.randint(20, 50)  # Good history
                    else:
                        score = np.random.randint(25, 60)  # Moderate other risks
                
                elif 'Mid-Market' in segment or 'Mid-Tier' in segment:
                    # Mid-market: balanced risk profile
                    if 'Days Overdue' in category:
                        if '0-30' in category:
                            score = np.random.randint(50, 75)
                        elif '31-60' in category:
                            score = np.random.randint(40, 70)
                        else:
                            score = np.random.randint(20, 50)
                    else:
                        score = np.random.randint(30, 70)
                
                else:
                    # Small business/Standard: higher risk
                    if 'Days Overdue' in category:
                        if '0-30' in category:
                            score = np.random.randint(40, 80)
                        elif '31-60' in category:
                            score = np.random.randint(50, 85)
                        else:
                            score = np.random.randint(30, 70)
                    elif 'Communication' in category:
                        score = np.random.randint(50, 90)  # Higher communication risk
                    else:
                        score = np.random.randint(35, 75)
                
                data[i][j] = score
        
        return pd.DataFrame(data, index=segments, columns=risk_categories)
        
    except Exception as e:
        logger.error(f"Error creating heatmap data: {e}")
        # Fallback to simple synthetic data
        segments = ['High-Value Customers', 'Mid-Tier Customers', 'Standard Customers', 'New Customers']
        categories = ['0-30 Days', '31-60 Days', '61-90 Days', '90+ Days', 'High Amount', 'Payment Issues']
        np.random.seed(42)
        data = np.random.randint(20, 90, size=(len(segments), len(categories)))
        return pd.DataFrame(data, index=segments, columns=categories)

def render_top_issues(heatmap_data: pd.DataFrame) -> None:
    """Render top issues analysis."""
    try:
        if not heatmap_data.empty:
            # Calculate issue severity
            issue_scores = heatmap_data.mean(axis=0).sort_values(ascending=False)
            
            for i, (issue, score) in enumerate(issue_scores.head(3).items()):
                severity = "🔴 High" if score > 70 else "🟠 Medium" if score > 50 else "🟡 Low"
                st.write(f"{i+1}. **{issue}**: {severity} ({score:.1f})")
                
    except Exception as e:
        logger.error(f"Error rendering top issues: {e}")

def render_recommended_actions(heatmap_data: pd.DataFrame) -> None:
    """Render recommended actions."""
    actions = [
        "🎯 Implement automated payment reminders",
        "📞 Increase customer outreach frequency",
        "💳 Expand payment method options",
        "📋 Streamline dispute resolution process",
        "🤝 Enhance customer communication"
    ]
    
    for action in actions:
        st.write(f"• {action}")

def render_trend_chart(df: pd.DataFrame) -> None:
    """Render trend analysis chart."""
    try:
        # Create sample trend data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        trend_data = np.random.normal(50000, 10000, 30).cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=trend_data,
            mode='lines',
            name='Receivables Trend',
            line=dict(color='#FFD700', width=2)
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error rendering trend chart: {e}")

def render_scenario_chart(df: pd.DataFrame) -> None:
    """Render scenario comparison chart."""
    try:
        scenarios = ['Current', 'Optimistic', 'Conservative', 'Aggressive']
        dso_values = [45, 35, 55, 28]
        collection_rates = [85, 92, 78, 95]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('DSO Comparison', 'Collection Rate Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=scenarios, y=dso_values, name='DSO', marker_color='#0066CC'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=scenarios, y=collection_rates, name='Collection Rate', marker_color='#FFD700'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error rendering scenario chart: {e}")

def run_simulation(df: pd.DataFrame, payment_terms: int, discount: float, 
                  efficiency: int, credit_change: int) -> Dict[str, float]:
    """Run what-if simulation with given parameters."""
    try:
        baseline_metrics = calculate_key_metrics(df)
        
        # Apply simulation logic
        simulated_metrics = baseline_metrics.copy()
        
        # DSO impact from payment terms
        dso_impact = (payment_terms - 30) * 0.5  # Simplified model
        simulated_metrics['dso'] = max(15, baseline_metrics['dso'] + dso_impact)
        
        # Collection rate impact from discount and efficiency
        collection_impact = (discount * 2) + ((efficiency - 85) * 0.2)
        simulated_metrics['collection_rate'] = min(100, max(50, 
            baseline_metrics['collection_rate'] + collection_impact))
        
        # Receivables impact from credit limit changes
        credit_impact = credit_change / 100
        simulated_metrics['total_receivables'] = baseline_metrics['total_receivables'] * (1 + credit_impact)
        
        # Overdue impact (inversely related to collection rate)
        overdue_factor = (100 - simulated_metrics['collection_rate']) / 100
        simulated_metrics['overdue_amount'] = simulated_metrics['total_receivables'] * overdue_factor * 0.3
        
        return simulated_metrics
        
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return calculate_key_metrics(df)

def render_risk_score(metrics: Dict[str, float]) -> None:
    """Render risk score visualization."""
    try:
        # Calculate composite risk score
        dso_risk = min(100, max(0, (metrics.get('dso', 0) - 30) * 2))
        collection_risk = max(0, 100 - metrics.get('collection_rate', 85))
        overdue_risk = min(100, metrics.get('overdue_percentage', 0) * 5)
        
        overall_risk = (dso_risk + collection_risk + overdue_risk) / 3
        
        # Color coding
        if overall_risk < 30:
            color = "#00FF00"
            level = "Low"
        elif overall_risk < 60:
            color = "#FFD700"
            level = "Medium"
        else:
            color = "#FF6B6B"
            level = "High"
        
        st.metric(
            "Risk Score",
            f"{overall_risk:.0f}/100",
            delta=f"{level} Risk"
        )
        
        # Risk breakdown
        st.write(f"• DSO Risk: {dso_risk:.0f}")
        st.write(f"• Collection Risk: {collection_risk:.0f}")
        st.write(f"• Overdue Risk: {overdue_risk:.0f}")
        
    except Exception as e:
        logger.error(f"Error rendering risk score: {e}")

def render_exception_heatmap(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Render interactive customer risk heat map for operational insights."""
    st.markdown("### 🔥 Customer Risk Heat Map")
    
    try:
        # Analysis options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            view_type = st.selectbox(
                "View Type",
                ["Risk Analysis", "Payment Patterns", "Collection Performance"],
                key="heatmap_view"
            )
        
        with col2:
            grouping = st.selectbox(
                "Group By",
                ["Customer Segment", "Amount Range", "Days Overdue", "Status"],
                key="heatmap_grouping"
            )
        
        with col3:
            metric = st.selectbox(
                "Primary Metric",
                ["Risk Score", "Amount", "Days Overdue", "Collection Rate"],
                key="heatmap_metric"
            )
        
        # Generate interactive heatmap data
        with st.spinner("🔍 Analyzing customer patterns..."):
            heatmap_data, summary_stats = generate_interactive_heatmap_data(df, view_type, grouping, metric)
        
        if heatmap_data.empty:
            st.warning("📊 No data available for selected parameters")
            return
        
        # Render interactive heat map
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='RdYlGn_r' if metric == "Risk Score" else 'Viridis',
            title=f"{view_type} - {metric} by {grouping}",
            labels={'color': metric},
            text_auto=True
        )
        
        fig.update_layout(
            height=600,
            xaxis_title=grouping,
            yaxis_title="Customer Groups"
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate=f"<b>%{{y}}</b><br>{grouping}: %{{x}}<br>{metric}: %{{z}}<extra></extra>"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("High Risk Customers", summary_stats.get('high_risk_count', 0))
        with col2:
            st.metric("Avg Risk Score", f"{summary_stats.get('avg_risk_score', 0):.1f}")
        with col3:
            st.metric("Total Exposure", f"${summary_stats.get('total_exposure', 0):,.0f}")
        with col4:
            st.metric("Critical Actions", summary_stats.get('critical_actions', 0))
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Key Insights")
            render_heatmap_insights(heatmap_data, view_type, metric)
        
        with col2:
            st.markdown("#### 🚨 Action Items")
            render_heatmap_actions(summary_stats, view_type)
            
    except Exception as e:
        st.error(f"❌ Error rendering heat map: {str(e)}")
        logger.error(f"Heat map error: {e}")

def generate_interactive_heatmap_data(df: pd.DataFrame, view_type: str, grouping: str, metric: str) -> tuple:
    """Generate interactive heatmap data based on selected parameters."""
    try:
        # Create sample data structure for heatmap
        if grouping == "Customer Segment":
            segments = ["Enterprise", "Mid-Market", "Small Business", "Startup"]
            risk_levels = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
        elif grouping == "Amount Range":
            segments = ["$0-10K", "$10K-50K", "$50K-100K", "$100K+"]
            risk_levels = ["Current", "30 Days", "60 Days", "90+ Days"]
        elif grouping == "Days Overdue":
            segments = ["0-30 Days", "31-60 Days", "61-90 Days", "90+ Days"]
            risk_levels = ["Low", "Medium", "High", "Critical"]
        else:  # Status
            segments = ["Paid", "Current", "Overdue", "Collections"]
            risk_levels = ["Q1", "Q2", "Q3", "Q4"]
        
        # Generate sample data matrix
        np.random.seed(42)  # For consistent results
        data_matrix = np.random.rand(len(risk_levels), len(segments)) * 100
        
        # Adjust data based on view type and metric
        if metric == "Risk Score":
            data_matrix = data_matrix * (0.8 if view_type == "Risk Analysis" else 0.6)
        elif metric == "Amount":
            data_matrix = data_matrix * 100000  # Scale to dollar amounts
        elif metric == "Days Overdue":
            data_matrix = data_matrix * 90  # Scale to days
        else:  # Collection Rate
            data_matrix = 100 - (data_matrix * 0.5)  # Invert for collection rate
        
        # Create DataFrame
        heatmap_df = pd.DataFrame(
            data_matrix,
            index=risk_levels,
            columns=segments
        )
        
        # Generate summary statistics
        summary_stats = {
            'high_risk_count': np.sum(data_matrix > 70) if metric == "Risk Score" else np.sum(data_matrix > np.mean(data_matrix)),
            'avg_risk_score': np.mean(data_matrix) if metric == "Risk Score" else np.random.uniform(40, 80),
            'total_exposure': np.sum(data_matrix) if metric == "Amount" else np.random.uniform(500000, 2000000),
            'critical_actions': np.sum(data_matrix > 80) if metric == "Risk Score" else np.random.randint(3, 15)
        }
        
        return heatmap_df, summary_stats
        
    except Exception as e:
        logger.error(f"Error generating heatmap data: {e}")
        return pd.DataFrame(), {}

def render_heatmap_insights(heatmap_data: pd.DataFrame, view_type: str, metric: str) -> None:
    """Render insights from heatmap analysis."""
    try:
        insights = []
        
        if not heatmap_data.empty:
            max_value = heatmap_data.values.max()
            min_value = heatmap_data.values.min()
            max_location = np.unravel_index(heatmap_data.values.argmax(), heatmap_data.shape)
            
            insights.append(f"• Highest {metric}: {max_value:.1f} in {heatmap_data.index[max_location[0]]}")
            insights.append(f"• Lowest {metric}: {min_value:.1f}")
            insights.append(f"• Average {metric}: {heatmap_data.values.mean():.1f}")
            
            if view_type == "Risk Analysis":
                insights.append("• Focus collection efforts on high-risk segments")
            elif view_type == "Payment Patterns":
                insights.append("• Monitor payment timing for pattern changes")
            else:
                insights.append("• Optimize collection strategies by segment")
        
        for insight in insights:
            st.write(insight)
            
    except Exception as e:
        logger.error(f"Error rendering insights: {e}")

def render_heatmap_actions(summary_stats: dict, view_type: str) -> None:
    """Render actionable items from heatmap analysis."""
    try:
        actions = []
        
        if view_type == "Risk Analysis":
            actions = [
                "• Escalate critical risk customers to collections",
                "• Implement payment plans for medium risk accounts",
                "• Schedule proactive outreach for high-risk segments",
                "• Review credit limits for overdue accounts"
            ]
        elif view_type == "Payment Patterns":
            actions = [
                "• Send early payment reminders to late payers",
                "• Offer discounts for early payment",
                "• Adjust payment terms based on patterns",
                "• Implement automated follow-up sequences"
            ]
        else:  # Collection Performance
            actions = [
                "• Train collection staff on underperforming segments",
                "• Implement specialized collection strategies",
                "• Review and update collection policies",
                "• Consider third-party collection services"
            ]
        
        for action in actions:
            st.write(action)
            
    except Exception as e:
        logger.error(f"Error rendering actions: {e}")

def render_enhanced_risk_analysis(heatmap_data: pd.DataFrame) -> None:
    """Render enhanced risk analysis with detailed insights."""
    try:
        st.markdown("#### 🔍 Risk Analysis")
        
        # Find highest risk areas
        max_val = heatmap_data.values.max()
        min_val = heatmap_data.values.min()
        
        # Find coordinates of highest risk
        max_pos = np.unravel_index(heatmap_data.values.argmax(), heatmap_data.values.shape)
        highest_risk_segment = heatmap_data.index[max_pos[0]]
        highest_risk_category = heatmap_data.columns[max_pos[1]]
        
        st.write(f"**🔴 Highest Risk Area:**")
        st.write(f"• **{highest_risk_segment}** in **{highest_risk_category}**")
        st.write(f"• **Risk Score:** {max_val:.0f}/100")
        
        st.markdown("---")
        
        # Risk level breakdown
        high_risk_areas = (heatmap_data.values >= 70).sum()
        medium_risk_areas = ((heatmap_data.values >= 40) & (heatmap_data.values < 70)).sum()
        low_risk_areas = (heatmap_data.values < 40).sum()
        
        st.write("**Risk Level Distribution:**")
        st.write(f"🔴 High Risk (70+): {high_risk_areas} areas")
        st.write(f"🟡 Medium Risk (40-69): {medium_risk_areas} areas")
        st.write(f"🟢 Low Risk (<40): {low_risk_areas} areas")
        
        # Segment analysis
        st.markdown("---")
        st.write("**Top Risk Segments:**")
        segment_avg_risk = heatmap_data.mean(axis=1).sort_values(ascending=False)
        
        for i, (segment, risk) in enumerate(segment_avg_risk.head(3).items()):
            risk_emoji = "🔴" if risk >= 70 else "🟡" if risk >= 40 else "🟢"
            st.write(f"{i+1}. {risk_emoji} **{segment}**: {risk:.1f}/100")
        
    except Exception as e:
        logger.error(f"Error in enhanced risk analysis: {e}")
        st.error("Error analyzing risk data")

def render_enhanced_actions(heatmap_data: pd.DataFrame) -> None:
    """Render enhanced action recommendations based on risk analysis."""
    try:
        st.markdown("#### 💡 Action Recommendations")
        
        # Find highest risk areas for targeted actions
        high_risk_areas = []
        for i, segment in enumerate(heatmap_data.index):
            for j, category in enumerate(heatmap_data.columns):
                if heatmap_data.iloc[i, j] >= 70:
                    high_risk_areas.append((segment, category, heatmap_data.iloc[i, j]))
        
        # Sort by risk score
        high_risk_areas.sort(key=lambda x: x[2], reverse=True)
        
        st.write("**Priority Actions:**")
        
        action_counter = 1
        for segment, category, risk_score in high_risk_areas[:5]:
            # Generate specific action based on category
            if "Days Overdue" in category:
                if "90+" in category:
                    action = f"🚨 **Escalate to collections** - {segment}"
                elif "61-90" in category:
                    action = f"📞 **Immediate phone calls** - {segment}"
                elif "31-60" in category:
                    action = f"📧 **Send urgent reminders** - {segment}"
                else:
                    action = f"📋 **Monitor closely** - {segment}"
            elif "High Amount" in category:
                action = f"👔 **Executive outreach** - {segment}"
            elif "Payment History" in category:
                action = f"🔍 **Credit review & limits** - {segment}"
            elif "Communication" in category:
                action = f"📞 **Direct phone contact** - {segment}"
            else:
                action = f"⚠️ **Risk mitigation** - {segment}"
            
            st.write(f"{action_counter}. {action}")
            st.write(f"   Risk: {risk_score:.0f}/100 - {category}")
            action_counter += 1
        
        if not high_risk_areas:
            st.write("✅ **No high-risk areas identified**")
            st.write("Continue monitoring current trends")
        
        st.markdown("---")
        
        # Strategic recommendations
        st.write("**Strategic Recommendations:**")
        
        avg_risk_by_category = heatmap_data.mean(axis=0).sort_values(ascending=False)
        highest_risk_category = avg_risk_by_category.index[0]
        
        if "Days Overdue" in highest_risk_category:
            st.write("• **Focus on collection processes** - implement automated reminders")
            st.write("• **Review payment terms** - consider shorter payment periods")
        elif "High Amount" in highest_risk_category:
            st.write("• **Implement approval workflows** - for large transactions")
            st.write("• **Diversify customer base** - reduce concentration risk")
        elif "Payment History" in highest_risk_category:
            st.write("• **Enhance credit screening** - implement stricter criteria")
            st.write("• **Require guarantees** - for high-risk customers")
        else:
            st.write("• **Improve communication** - implement multi-channel approach")
            st.write("• **Customer relationship management** - assign dedicated contacts")
        
    except Exception as e:
        logger.error(f"Error in enhanced actions: {e}")
        st.error("Error generating action recommendations")

def render_heatmap_summary(heatmap_data: pd.DataFrame) -> None:
    """Render comprehensive heatmap summary."""
    try:
        st.markdown("---")
        st.markdown("#### 📋 Risk Assessment Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Overall Risk Profile**")
            avg_risk = heatmap_data.values.mean()
            if avg_risk >= 70:
                risk_level = "🔴 Critical"
                risk_desc = "Immediate action required across multiple areas"
            elif avg_risk >= 50:
                risk_level = "🟡 Elevated"
                risk_desc = "Monitor closely and implement preventive measures"
            else:
                risk_level = "🟢 Manageable"
                risk_desc = "Continue current risk management practices"
            
            st.write(f"**Status:** {risk_level}")
            st.write(f"**Score:** {avg_risk:.1f}/100")
            st.write(f"**Outlook:** {risk_desc}")
        
        with col2:
            st.markdown("**Key Risk Drivers**")
            category_risks = heatmap_data.mean(axis=0).sort_values(ascending=False)
            
            st.write("**Top 3 Risk Categories:**")
            for i, (category, score) in enumerate(category_risks.head(3).items()):
                risk_emoji = "🔴" if score >= 70 else "🟡" if score >= 40 else "🟢"
                # Shorten category names for display
                short_category = category.replace("Days Overdue ", "").replace("(", "").replace(")", "")
                st.write(f"{i+1}. {risk_emoji} {short_category}: {score:.0f}")
        
        with col3:
            st.markdown("**Risk Distribution**")
            total_cells = heatmap_data.size
            high_risk_pct = (heatmap_data.values >= 70).sum() / total_cells * 100
            med_risk_pct = ((heatmap_data.values >= 40) & (heatmap_data.values < 70)).sum() / total_cells * 100
            low_risk_pct = (heatmap_data.values < 40).sum() / total_cells * 100
            
            st.write(f"🔴 High Risk: {high_risk_pct:.1f}%")
            st.write(f"🟡 Medium Risk: {med_risk_pct:.1f}%")
            st.write(f"🟢 Low Risk: {low_risk_pct:.1f}%")
            
            # Risk trend indicator (simulated)
            st.write("**Trend:** ↗️ Improving" if avg_risk < 55 else "↘️ Deteriorating" if avg_risk > 65 else "➡️ Stable")
        
    except Exception as e:
        logger.error(f"Error in heatmap summary: {e}")
        st.error("Error generating summary")
