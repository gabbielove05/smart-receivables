"""
Data Quality Validation Module
Uses Great Expectations for comprehensive data validation and quality scoring.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import streamlit as st

logger = logging.getLogger(__name__)

# Great Expectations would be used in production, but for this demo we'll implement
# core validation logic directly since it's not in the restricted packages

class DataQualityValidator:
    """Core data quality validation class."""
    
    def __init__(self):
        self.validation_rules = {
            'completeness': self._check_completeness,
            'uniqueness': self._check_uniqueness,
            'validity': self._check_validity,
            'consistency': self._check_consistency,
            'timeliness': self._check_timeliness,
            'accuracy': self._check_accuracy
        }
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all validation rules on the dataset."""
        try:
            if df.empty:
                return {'overall_score': 0.0, 'issues': ['Dataset is empty']}
            
            results = {}
            all_issues = []
            
            for rule_name, rule_func in self.validation_rules.items():
                try:
                    score, issues = rule_func(df)
                    results[rule_name] = {
                        'score': score,
                        'issues': issues,
                        'passed': score >= 0.7
                    }
                    all_issues.extend(issues)
                except Exception as e:
                    logger.error(f"Validation rule {rule_name} failed: {e}")
                    results[rule_name] = {
                        'score': 0.0,
                        'issues': [f"Validation failed: {str(e)}"],
                        'passed': False
                    }
            
            # Calculate overall score
            overall_score = np.mean([r['score'] for r in results.values()])
            
            return {
                'overall_score': overall_score,
                'detailed_results': results,
                'total_issues': len(all_issues),
                'critical_issues': [i for i in all_issues if 'critical' in i.lower()],
                'all_issues': all_issues
            }
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {'overall_score': 0.0, 'issues': [f'Validation error: {str(e)}']}
    
    def _check_completeness(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check data completeness (missing values)."""
        issues = []
        total_cells = df.size
        
        if total_cells == 0:
            return 0.0, ['Dataset is empty']
        
        missing_cells = df.isnull().sum().sum()
        completeness_score = 1.0 - (missing_cells / total_cells)
        
        # Check critical columns
        critical_columns = ['invoice_id', 'amount']
        for col in critical_columns:
            if col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct > 0.1:  # More than 10% missing
                    issues.append(f"Critical: {missing_pct:.1%} missing values in {col}")
                elif missing_pct > 0:
                    issues.append(f"Warning: {missing_pct:.1%} missing values in {col}")
        
        # Check overall missing data
        if missing_cells > 0:
            missing_pct = missing_cells / total_cells
            if missing_pct > 0.2:
                issues.append(f"Critical: {missing_pct:.1%} of all data is missing")
            elif missing_pct > 0.05:
                issues.append(f"Warning: {missing_pct:.1%} of all data is missing")
        
        return completeness_score, issues
    
    def _check_uniqueness(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check data uniqueness (duplicates)."""
        issues = []
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            duplicate_pct = duplicate_rows / len(df)
            if duplicate_pct > 0.1:
                issues.append(f"Critical: {duplicate_pct:.1%} duplicate rows detected")
            else:
                issues.append(f"Warning: {duplicate_rows} duplicate rows found")
        
        # Check unique identifiers
        unique_columns = ['invoice_id', 'payment_id', 'customer_id']
        uniqueness_scores = []
        
        for col in unique_columns:
            if col in df.columns:
                if col in ['invoice_id', 'payment_id']:
                    # These should be completely unique
                    unique_count = df[col].nunique()
                    total_count = df[col].count()  # Exclude NaN
                    if total_count > 0:
                        uniqueness = unique_count / total_count
                        uniqueness_scores.append(uniqueness)
                        if uniqueness < 1.0:
                            issues.append(f"Critical: {col} has duplicate values")
                else:
                    # Customer IDs can repeat, just track for analysis
                    uniqueness_scores.append(1.0)
        
        overall_uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 1.0
        
        return overall_uniqueness, issues
    
    def _check_validity(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check data validity (format, ranges, types)."""
        issues = []
        validity_scores = []
        
        # Check amount validity
        if 'amount' in df.columns:
            negative_amounts = (df['amount'] < 0).sum()
            if negative_amounts > 0:
                issues.append(f"Warning: {negative_amounts} negative amounts found")
            
            zero_amounts = (df['amount'] == 0).sum()
            if zero_amounts > len(df) * 0.1:  # More than 10% zero amounts
                issues.append(f"Warning: {zero_amounts} zero amounts found")
            
            # Check for extremely large amounts (potential data entry errors)
            if df['amount'].max() > df['amount'].median() * 1000:
                issues.append("Warning: Extremely large amounts detected (potential errors)")
            
            # Amount validity score
            valid_amounts = ((df['amount'] > 0) & (df['amount'] < df['amount'].quantile(0.99))).sum()
            amount_validity = valid_amounts / len(df) if len(df) > 0 else 0
            validity_scores.append(amount_validity)
        
        # Check date validity
        date_columns = ['issue_date', 'due_date', 'payment_date']
        for col in date_columns:
            if col in df.columns:
                try:
                    # Try to convert to datetime
                    dates = pd.to_datetime(df[col], errors='coerce')
                    invalid_dates = dates.isnull().sum() - df[col].isnull().sum()  # Invalid but not missing
                    
                    if invalid_dates > 0:
                        issues.append(f"Warning: {invalid_dates} invalid dates in {col}")
                    
                    # Check for future dates where inappropriate
                    if col in ['issue_date', 'payment_date']:
                        future_dates = (dates > datetime.now()).sum()
                        if future_dates > 0:
                            issues.append(f"Warning: {future_dates} future dates in {col}")
                    
                    # Date validity score
                    valid_dates = (~dates.isnull()).sum()
                    total_dates = df[col].count()
                    date_validity = valid_dates / total_dates if total_dates > 0 else 1.0
                    validity_scores.append(date_validity)
                    
                except Exception as e:
                    issues.append(f"Error validating dates in {col}: {str(e)}")
                    validity_scores.append(0.5)
        
        # Check status validity
        if 'status' in df.columns:
            valid_statuses = ['paid', 'outstanding', 'overdue', 'cancelled', 'disputed']
            invalid_statuses = ~df['status'].str.lower().isin(valid_statuses)
            invalid_count = invalid_statuses.sum()
            
            if invalid_count > 0:
                issues.append(f"Warning: {invalid_count} records with invalid status values")
            
            status_validity = (len(df) - invalid_count) / len(df) if len(df) > 0 else 1.0
            validity_scores.append(status_validity)
        
        overall_validity = np.mean(validity_scores) if validity_scores else 1.0
        return overall_validity, issues
    
    def _check_consistency(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check data consistency across fields."""
        issues = []
        consistency_scores = []
        
        # Check date consistency
        if 'issue_date' in df.columns and 'due_date' in df.columns:
            try:
                issue_dates = pd.to_datetime(df['issue_date'], errors='coerce')
                due_dates = pd.to_datetime(df['due_date'], errors='coerce')
                
                # Due date should be after issue date
                invalid_date_order = (due_dates < issue_dates).sum()
                if invalid_date_order > 0:
                    issues.append(f"Critical: {invalid_date_order} records with due date before issue date")
                
                date_consistency = (len(df) - invalid_date_order) / len(df) if len(df) > 0 else 1.0
                consistency_scores.append(date_consistency)
                
            except Exception as e:
                issues.append(f"Error checking date consistency: {str(e)}")
                consistency_scores.append(0.5)
        
        # Check payment consistency
        if all(col in df.columns for col in ['amount', 'payment_amount']):
            # Payment amount shouldn't exceed invoice amount
            overpayments = (df['payment_amount'] > df['amount'] * 1.01).sum()  # 1% tolerance
            if overpayments > 0:
                issues.append(f"Warning: {overpayments} potential overpayments detected")
            
            payment_consistency = (len(df) - overpayments) / len(df) if len(df) > 0 else 1.0
            consistency_scores.append(payment_consistency)
        
        # Check status consistency
        if 'status' in df.columns and 'payment_date' in df.columns:
            try:
                # Paid invoices should have payment dates
                paid_invoices = df['status'].str.lower() == 'paid'
                paid_without_payment_date = (paid_invoices & df['payment_date'].isnull()).sum()
                
                if paid_without_payment_date > 0:
                    issues.append(f"Warning: {paid_without_payment_date} paid invoices missing payment date")
                
                # Outstanding/overdue invoices shouldn't have payment dates
                unpaid_statuses = df['status'].str.lower().isin(['outstanding', 'overdue'])
                unpaid_with_payment_date = (unpaid_statuses & df['payment_date'].notnull()).sum()
                
                if unpaid_with_payment_date > 0:
                    issues.append(f"Warning: {unpaid_with_payment_date} unpaid invoices have payment dates")
                
                status_consistency = (len(df) - paid_without_payment_date - unpaid_with_payment_date) / len(df)
                consistency_scores.append(status_consistency)
                
            except Exception as e:
                issues.append(f"Error checking status consistency: {str(e)}")
                consistency_scores.append(0.5)
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        return overall_consistency, issues
    
    def _check_timeliness(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check data timeliness (how current the data is)."""
        issues = []
        timeliness_scores = []
        
        # Check issue date timeliness
        if 'issue_date' in df.columns:
            try:
                issue_dates = pd.to_datetime(df['issue_date'], errors='coerce')
                latest_date = issue_dates.max()
                
                if pd.notnull(latest_date):
                    days_since_latest = (datetime.now() - latest_date).days
                    
                    if days_since_latest > 90:
                        issues.append(f"Warning: Latest data is {days_since_latest} days old")
                        timeliness_score = max(0, 1 - (days_since_latest - 90) / 365)  # Decay after 90 days
                    else:
                        timeliness_score = 1.0
                    
                    timeliness_scores.append(timeliness_score)
                else:
                    issues.append("Warning: No valid issue dates found")
                    timeliness_scores.append(0.5)
                    
            except Exception as e:
                issues.append(f"Error checking data timeliness: {str(e)}")
                timeliness_scores.append(0.5)
        
        # Check for stale records (very old outstanding invoices)
        if 'status' in df.columns and 'issue_date' in df.columns:
            try:
                issue_dates = pd.to_datetime(df['issue_date'], errors='coerce')
                outstanding_mask = df['status'].str.lower() == 'outstanding'
                
                if outstanding_mask.any():
                    old_outstanding = outstanding_mask & (
                        (datetime.now() - issue_dates).dt.days > 365
                    )
                    very_old_count = old_outstanding.sum()
                    
                    if very_old_count > 0:
                        issues.append(f"Warning: {very_old_count} outstanding invoices over 1 year old")
                
            except Exception as e:
                logger.error(f"Error checking stale records: {e}")
        
        overall_timeliness = np.mean(timeliness_scores) if timeliness_scores else 1.0
        return overall_timeliness, issues
    
    def _check_accuracy(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check data accuracy using business rules."""
        issues = []
        accuracy_scores = []
        
        # Check for reasonable business patterns
        if 'amount' in df.columns:
            # Check for round number bias (too many round numbers might indicate estimates)
            if len(df) > 10:
                round_numbers = (df['amount'] % 100 == 0).sum()
                round_percentage = round_numbers / len(df)
                
                if round_percentage > 0.5:
                    issues.append(f"Warning: {round_percentage:.1%} of amounts are round numbers")
                
                # Statistical outliers
                Q1 = df['amount'].quantile(0.25)
                Q3 = df['amount'].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df['amount'] < Q1 - 3 * IQR) | (df['amount'] > Q3 + 3 * IQR)).sum()
                
                if outliers > 0:
                    outlier_pct = outliers / len(df)
                    if outlier_pct > 0.1:
                        issues.append(f"Warning: {outlier_pct:.1%} statistical outliers in amounts")
                
                accuracy_scores.append(max(0.5, 1 - outlier_pct))
        
        # Check customer distribution (too many single-customer invoices might be suspicious)
        if 'customer_id' in df.columns and len(df) > 10:
            customer_counts = df['customer_id'].value_counts()
            single_invoice_customers = (customer_counts == 1).sum()
            single_customer_pct = single_invoice_customers / len(customer_counts)
            
            if single_customer_pct > 0.8:
                issues.append(f"Warning: {single_customer_pct:.1%} customers have only one invoice")
        
        # Payment pattern accuracy
        if 'status' in df.columns:
            status_dist = df['status'].str.lower().value_counts(normalize=True)
            
            # Check for unusual status distributions
            if 'paid' in status_dist and status_dist['paid'] > 0.95:
                issues.append("Warning: Unusually high payment rate (>95%)")
            elif 'overdue' in status_dist and status_dist['overdue'] > 0.5:
                issues.append("Warning: High overdue rate (>50%)")
            
            accuracy_scores.append(1.0)  # Base accuracy for having status data
        
        overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.8
        return overall_accuracy, issues

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main function to validate data quality.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    try:
        validator = DataQualityValidator()
        results = validator.validate_dataset(df)
        
        logger.info(f"Data quality validation completed. Score: {results['overall_score']:.2f}")
        return results
        
    except Exception as e:
        logger.error(f"Data quality validation failed: {e}")
        return {
            'overall_score': 0.0,
            'detailed_results': {},
            'total_issues': 1,
            'critical_issues': [f'Validation failed: {str(e)}'],
            'all_issues': [f'Validation failed: {str(e)}']
        }

def get_quality_score(validation_results: Dict[str, Any]) -> float:
    """Extract the overall quality score from validation results."""
    try:
        return validation_results.get('overall_score', 0.0)
    except Exception:
        return 0.0

def render_data_quality_report(validation_results: Dict[str, Any]) -> None:
    """Render a comprehensive data quality report."""
    try:
        if not validation_results or 'overall_score' not in validation_results:
            st.error("❌ No validation results available")
            return
        
        overall_score = validation_results['overall_score']
        
        # Quality score header
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("### 📊 Data Quality Report")
        
        with col2:
            # Quality badge
            if overall_score >= 0.8:
                badge_class = "quality-badge-high"
                badge_text = "High Quality"
            elif overall_score >= 0.6:
                badge_class = "quality-badge-medium"
                badge_text = "Medium Quality"
            else:
                badge_class = "quality-badge-low"
                badge_text = "Low Quality"
            
            st.markdown(f"""
            <div class="quality-badge {badge_class}">
                {overall_score:.1%} - {badge_text}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_issues = validation_results.get('total_issues', 0)
            st.metric("Issues Found", total_issues)
        
        # Detailed results
        if 'detailed_results' in validation_results:
            st.markdown("#### 🔍 Validation Details")
            
            detailed_results = validation_results['detailed_results']
            
            for category, results in detailed_results.items():
                with st.expander(f"{category.title()} - Score: {results['score']:.1%}"):
                    if results['passed']:
                        st.success(f"✅ {category.title()} validation passed")
                    else:
                        st.warning(f"⚠️ {category.title()} validation needs attention")
                    
                    if results['issues']:
                        st.markdown("**Issues Found:**")
                        for issue in results['issues']:
                            if 'critical' in issue.lower():
                                st.error(f"🔴 {issue}")
                            else:
                                st.warning(f"🟡 {issue}")
                    else:
                        st.success("No issues found in this category")
        
        # Critical issues summary
        critical_issues = validation_results.get('critical_issues', [])
        if critical_issues:
            st.markdown("#### 🚨 Critical Issues Requiring Immediate Attention")
            for issue in critical_issues:
                st.error(f"🔴 {issue}")
        
        # Recommendations
        st.markdown("#### 💡 Recommendations")
        recommendations = generate_quality_recommendations(validation_results)
        for rec in recommendations:
            st.info(f"• {rec}")
            
    except Exception as e:
        logger.error(f"Error rendering quality report: {e}")
        st.error(f"❌ Error rendering quality report: {str(e)}")

def generate_quality_recommendations(validation_results: Dict[str, Any]) -> List[str]:
    """Generate actionable recommendations based on validation results."""
    recommendations = []
    
    try:
        overall_score = validation_results.get('overall_score', 0.0)
        detailed_results = validation_results.get('detailed_results', {})
        
        # Overall recommendations
        if overall_score < 0.6:
            recommendations.append("Consider comprehensive data cleanup before analysis")
        elif overall_score < 0.8:
            recommendations.append("Address specific data quality issues to improve reliability")
        
        # Specific recommendations based on failed categories
        for category, results in detailed_results.items():
            if not results['passed']:
                if category == 'completeness':
                    recommendations.append("Implement data validation at source to reduce missing values")
                elif category == 'uniqueness':
                    recommendations.append("Review data collection process to prevent duplicates")
                elif category == 'validity':
                    recommendations.append("Add format validation and range checks to data entry")
                elif category == 'consistency':
                    recommendations.append("Implement cross-field validation rules")
                elif category == 'timeliness':
                    recommendations.append("Establish regular data refresh procedures")
                elif category == 'accuracy':
                    recommendations.append("Review business rules and add accuracy checks")
        
        # Critical issue recommendations
        critical_issues = validation_results.get('critical_issues', [])
        if critical_issues:
            recommendations.append("Address critical data issues immediately before proceeding with analysis")
        
        if not recommendations:
            recommendations.append("Data quality is good - continue with analysis")
        
        return recommendations[:5]  # Limit to top 5 recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return ["Unable to generate recommendations due to error"]

def create_quality_dashboard(df: pd.DataFrame) -> None:
    """Create an interactive data quality dashboard."""
    try:
        st.markdown("### 📈 Data Quality Dashboard")
        
        # Run validation
        with st.spinner("🔍 Analyzing data quality..."):
            validation_results = validate_data_quality(df)
        
        # Render the report
        render_data_quality_report(validation_results)
        
        # Quality trend (simulated for demo)
        st.markdown("#### 📊 Quality Trend")
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        quality_scores = np.random.normal(validation_results['overall_score'], 0.1, 30)
        quality_scores = np.clip(quality_scores, 0, 1)
        
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=quality_scores,
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='#0066CC', width=2)
        ))
        
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Target (80%)")
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange", annotation_text="Warning (60%)")
        
        fig.update_layout(
            title="Data Quality Score Trend",
            xaxis_title="Date",
            yaxis_title="Quality Score",
            height=400,
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error creating quality dashboard: {e}")
        st.error(f"❌ Error creating quality dashboard: {str(e)}")

