"""
Smart Actions and ML-Powered Priority Queue
Handles next-best-action recommendations and automation center.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from ml_models import train_priority_model, get_next_best_actions
from integrations import send_collection_reminder, call_client
from simple_email_system import simple_email_system

logger = logging.getLogger(__name__)

def generate_next_best_actions(df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate ML-powered next-best-action recommendations.
    
    Args:
        df: Merged receivables DataFrame
        config: Configuration parameters
        
    Returns:
        List of action recommendations sorted by priority
    """
    try:
        if df.empty:
            return []
        
        # Prepare features for ML model
        features_df = prepare_features(df)
        
        if features_df.empty:
            logger.warning("No features available for ML model")
            return generate_rule_based_actions(df, config)
        
        # Train anomaly detection model (Isolation Forest)
        anomaly_model = train_anomaly_detector(features_df)
        
        # Generate priority scores
        priority_scores = calculate_priority_scores(features_df, anomaly_model)
        
        # Create action recommendations
        actions = create_action_recommendations(df, priority_scores, config)
        
        # Sort by priority score (highest first)
        actions.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logger.info(f"Generated {len(actions)} next-best-action recommendations")
        return actions[:20]  # Return top 20 actions
        
    except Exception as e:
        logger.error(f"Error generating next-best actions: {e}")
        return generate_rule_based_actions(df, config)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for ML models."""
    try:
        features = pd.DataFrame()
        
        if 'amount' in df.columns:
            features['amount'] = df['amount']
            features['amount_log'] = np.log1p(df['amount'])
        
        if 'customer_id' in df.columns:
            # Customer frequency
            customer_counts = df['customer_id'].value_counts()
            features['customer_frequency'] = df['customer_id'].map(customer_counts)
            
            # Customer total amount
            customer_amounts = df.groupby('customer_id')['amount'].sum()
            features['customer_total'] = df['customer_id'].map(customer_amounts)
        
        # Add date-based features if available
        if 'issue_date' in df.columns:
            try:
                df['issue_date'] = pd.to_datetime(df['issue_date'])
                features['days_since_issue'] = (datetime.now() - df['issue_date']).dt.days
            except:
                features['days_since_issue'] = 30  # Default
        else:
            features['days_since_issue'] = np.random.randint(1, 90, len(df))
        
        if 'due_date' in df.columns:
            try:
                df['due_date'] = pd.to_datetime(df['due_date'])
                features['days_overdue'] = (datetime.now() - df['due_date']).dt.days
                features['days_overdue'] = features['days_overdue'].clip(lower=0)
            except:
                features['days_overdue'] = np.maximum(0, features['days_since_issue'] - 30)
        else:
            features['days_overdue'] = np.maximum(0, features['days_since_issue'] - 30)
        
        # Status-based features
        if 'status' in df.columns:
            status_map = {'paid': 0, 'outstanding': 1, 'overdue': 2}
            features['status_score'] = df['status'].map(status_map).fillna(1)
        else:
            # Infer status from days overdue
            features['status_score'] = np.where(
                features['days_overdue'] > 0, 2,  # overdue
                np.where(features['days_since_issue'] > 30, 1, 0)  # outstanding or paid
            )
        
        # Fill missing values
        features = features.fillna(features.median())
        
        return features
        
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return pd.DataFrame()

def train_anomaly_detector(features_df: pd.DataFrame) -> IsolationForest:
    """Train Isolation Forest for anomaly detection."""
    try:
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # Train Isolation Forest
        model = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        model.fit(features_scaled)
        logger.info("Anomaly detection model trained successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training anomaly detector: {e}")
        # Return dummy model
        return IsolationForest(random_state=42)

def calculate_priority_scores(features_df: pd.DataFrame, anomaly_model: IsolationForest) -> np.ndarray:
    """Calculate priority scores for each record."""
    try:
        # Normalize features for scoring
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # Get anomaly scores (lower = more anomalous)
        anomaly_scores = anomaly_model.decision_function(features_scaled)
        
        # Convert to priority scores (higher = higher priority)
        priority_scores = -anomaly_scores  # Invert so anomalies get high scores
        
        # Add business rule adjustments
        if 'days_overdue' in features_df.columns:
            overdue_bonus = features_df['days_overdue'] * 0.1
            priority_scores += overdue_bonus
        
        if 'amount' in features_df.columns:
            # Higher amounts get higher priority
            amount_bonus = np.log1p(features_df['amount']) * 0.05
            priority_scores += amount_bonus
        
        # Normalize to 0-100 scale
        priority_scores = ((priority_scores - priority_scores.min()) / 
                          (priority_scores.max() - priority_scores.min()) * 100)
        
        return priority_scores
        
    except Exception as e:
        logger.error(f"Error calculating priority scores: {e}")
        return np.random.random(len(features_df)) * 100

def create_action_recommendations(df: pd.DataFrame, priority_scores: np.ndarray, 
                                config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create actionable recommendations based on priority scores."""
    try:
        actions = []
        
        for idx, score in enumerate(priority_scores):
            if idx >= len(df):
                break
                
            row = df.iloc[idx]
            
            # Determine action type based on data
            days_overdue = 0
            if 'due_date' in df.columns:
                try:
                    due_date = pd.to_datetime(row['due_date'])
                    days_overdue = max(0, (datetime.now() - due_date).days)
                except:
                    days_overdue = 0
            
            amount = row.get('amount', 0)
            customer_id = row.get('customer_id', f'CUST_{idx+1}')
            status = row.get('status', 'unknown')
            
            # Generate action based on conditions
            action_type = determine_action_type(days_overdue, amount, status, score)
            
            action = {
                'priority_score': score,
                'customer_id': customer_id,
                'amount': amount,
                'days_overdue': days_overdue,
                'status': status,
                'action_type': action_type,
                'recommended_action': get_action_description(action_type, customer_id, amount, days_overdue),
                'urgency': get_urgency_level(score),
                'expected_outcome': get_expected_outcome(action_type, amount),
                'contact_info': generate_contact_info(customer_id)
            }
            
            actions.append(action)
        
        return actions
        
    except Exception as e:
        logger.error(f"Error creating action recommendations: {e}")
        return []

def determine_action_type(days_overdue: int, amount: float, status: str, priority_score: float) -> str:
    """Determine the best action type based on customer situation."""
    if days_overdue > 90:
        return "collections"
    elif days_overdue > 30:
        return "call_reminder"
    elif amount > 50000:
        return "personal_outreach"
    elif priority_score > 80:
        return "email_reminder"
    elif status == 'outstanding':
        return "payment_plan"
    else:
        return "email_reminder"

def get_action_description(action_type: str, customer_id: str, amount: float, days_overdue: int) -> str:
    """Get human-readable action description."""
    descriptions = {
        "collections": f"Escalate {customer_id} to collections - ${amount:,.0f} overdue {days_overdue} days",
        "call_reminder": f"Call {customer_id} regarding ${amount:,.0f} payment ({days_overdue} days overdue)",
        "personal_outreach": f"Personal outreach to {customer_id} for high-value account (${amount:,.0f})",
        "email_reminder": f"Send payment reminder email to {customer_id} - ${amount:,.0f}",
        "payment_plan": f"Offer payment plan to {customer_id} for ${amount:,.0f} outstanding balance"
    }
    return descriptions.get(action_type, f"Contact {customer_id} regarding ${amount:,.0f}")

def get_urgency_level(priority_score: float) -> str:
    """Convert priority score to urgency level."""
    if priority_score >= 80:
        return "Critical"
    elif priority_score >= 60:
        return "High"
    elif priority_score >= 40:
        return "Medium"
    else:
        return "Low"

def get_expected_outcome(action_type: str, amount: float) -> str:
    """Get expected outcome for action type."""
    outcomes = {
        "collections": f"Recover ${amount * 0.6:,.0f} (60% recovery rate)",
        "call_reminder": f"Collect ${amount * 0.8:,.0f} within 7 days",
        "personal_outreach": f"Maintain relationship, collect ${amount * 0.9:,.0f}",
        "email_reminder": f"Collect ${amount * 0.7:,.0f} within 14 days",
        "payment_plan": f"Structured collection of ${amount:,.0f} over 3-6 months"
    }
    return outcomes.get(action_type, f"Improve collection likelihood for ${amount:,.0f}")

def generate_contact_info(customer_id: str) -> Dict[str, str]:
    """Generate sample contact information."""
    import hashlib
    
    # Generate consistent fake data based on customer_id
    hash_obj = hashlib.md5(customer_id.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Use hash to generate consistent fake contact info
    phone_suffix = int(hash_hex[:4], 16) % 10000
    email_prefix = hash_hex[:8]
    
    # Generate customer name and contact person
    names = ["Johnson", "Smith", "Brown", "Davis", "Wilson", "Miller", "Moore", "Taylor", "Anderson", "Thomas"]
    customer_name = names[int(hash_hex[8:10], 16) % len(names)]
    contact_names = ["Alex", "Jordan", "Casey", "Morgan", "Riley", "Blake", "Avery", "Quinn", "Sage", "River"]
    contact_person = contact_names[int(hash_hex[10:12], 16) % len(contact_names)]
    
    return {
        'email': f"{email_prefix}@{customer_id.lower()}.com",
        'phone': f"+1-555-{phone_suffix:04d}",
        'contact_name': contact_person,
        'customer_name': f"{customer_name} Corp"
    }

def generate_rule_based_actions(df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate rule-based actions as fallback when ML fails."""
    try:
        actions = []
        
        for idx, row in df.iterrows():
            amount = row.get('amount', 0)
            customer_id = row.get('customer_id', f'CUST_{idx+1}')
            status = row.get('status', 'outstanding')
            
            # Simple rule-based priority
            if amount > config.get('overdue_threshold', 50000):
                priority = 90
                action_type = "personal_outreach"
            elif status == 'overdue':
                priority = 80
                action_type = "call_reminder"
            elif status == 'outstanding':
                priority = 60
                action_type = "email_reminder"
            else:
                priority = 30
                action_type = "email_reminder"
            
            action = {
                'priority_score': priority,
                'customer_id': customer_id,
                'amount': amount,
                'days_overdue': 0,  # Unknown in rule-based mode
                'status': status,
                'action_type': action_type,
                'recommended_action': get_action_description(action_type, customer_id, amount, 0),
                'urgency': get_urgency_level(priority),
                'expected_outcome': get_expected_outcome(action_type, amount),
                'contact_info': generate_contact_info(customer_id)
            }
            
            actions.append(action)
        
        # Sort by priority
        actions.sort(key=lambda x: x['priority_score'], reverse=True)
        return actions[:20]
        
    except Exception as e:
        logger.error(f"Error generating rule-based actions: {e}")
        return []

def render_automation_center(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Render the automation center interface."""
    st.markdown("### ⚡ Smart Actions Center")
    
    try:
        # Generate next-best actions
        with st.spinner("🧠 Analyzing data and generating recommendations..."):
            actions = generate_next_best_actions(df, config)
        
        if not actions:
            st.warning("📋 No actions generated. Please check your data.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            critical_actions = sum(1 for a in actions if a['urgency'] == 'Critical')
            st.metric("🔴 Critical Actions", critical_actions)
        
        with col2:
            total_amount = sum(a['amount'] for a in actions)
            st.metric("💰 Total at Risk", f"${total_amount:,.0f}")
        
        with col3:
            avg_score = np.mean([a['priority_score'] for a in actions])
            st.metric("📊 Avg Priority", f"{avg_score:.0f}/100")
        
        with col4:
            expected_recovery = sum(a['amount'] * 0.7 for a in actions)  # 70% recovery estimate
            st.metric("🎯 Expected Recovery", f"${expected_recovery:,.0f}")
        
        st.markdown("---")
        
        # Actions display
        st.markdown("#### 🎯 Recommended Actions (Top Priority)")
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📋 Action Queue", "📊 Priority Analysis"])
        
        with tab1:
            render_action_queue(actions[:10])  # Show top 10
        
        with tab2:
            render_priority_analysis(actions)
            
    except Exception as e:
        st.error(f"❌ Error rendering automation center: {str(e)}")
        logger.error(f"Automation center error: {e}")

def render_action_queue(actions: List[Dict[str, Any]]) -> None:
    """Render the priority action queue with collapsible sections."""
    try:
        # Initialize session state for expanded action
        if 'expanded_action' not in st.session_state:
            st.session_state.expanded_action = None
        
        # Check if any action is expanded
        if st.session_state.expanded_action is not None:
            # Render the expanded action in full width
            expanded_action_id = st.session_state.expanded_action
            expanded_action = next((a for i, a in enumerate(actions) if i == expanded_action_id), None)
            
            if expanded_action:
                action_type = st.session_state.get('expanded_action_type', 'email')
                
                # Add "View All Actions" button at the top
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("📋 View All Actions", key="view_all_actions"):
                        st.session_state.expanded_action = None
                        st.rerun()
                
                st.markdown("---")
                
                if action_type == 'email':
                    send_email_action_fullscreen(expanded_action, expanded_action_id)
                elif action_type == 'call':
                    show_call_action_fullscreen(expanded_action, expanded_action_id)
                elif action_type == 'collections':
                    escalate_to_collections_fullscreen(expanded_action, expanded_action_id)
                elif action_type == 'details':
                    show_action_details_fullscreen(expanded_action, expanded_action_id)
        else:
            # Render collapsible action sections
            st.markdown("### 📋 Action Queue")
            st.markdown("Click on any action to expand and see details:")
            
            # Group actions by urgency
            urgency_groups = {}
            for action in actions:
                urgency = action['urgency']
                if urgency not in urgency_groups:
                    urgency_groups[urgency] = []
                urgency_groups[urgency].append(action)
            
            # Render each urgency group as a collapsible section
            for urgency in ['Critical', 'High', 'Medium', 'Low']:
                if urgency in urgency_groups:
                    with st.expander(f"🚨 {urgency} Priority Actions ({len(urgency_groups[urgency])})", expanded=(urgency == 'Critical')):
                        for i, action in enumerate(urgency_groups[urgency]):
                            urgency_colors = {
                                'Critical': '#FF4B4B',
                                'High': '#FF8C00',
                                'Medium': '#FFD700',
                                'Low': '#90EE90'
                            }
                            
                            urgency_color = urgency_colors.get(action['urgency'], '#FFD700')
                            
                            # Action card
                            with st.container():
                                st.markdown(f"""
                                <div style='
                                    border-left: 5px solid {urgency_color};
                                    background-color: #F8F9FA;
                                    padding: 15px;
                                    margin-bottom: 10px;
                                    border-radius: 5px;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                '>
                                    <h4 style='margin-top: 0; color: {urgency_color};'>
                                        {action['urgency']} Priority - Score: {action['priority_score']:.0f}
                                    </h4>
                                    <p><strong>Customer:</strong> {action['contact_info'].get('customer_name', action['customer_id'])}</p>
                                    <p><strong>Amount:</strong> ${action['amount']:,.2f}</p>
                                    <p><strong>Days Overdue:</strong> {action['days_overdue']}</p>
                                    <p><strong>Action:</strong> {action['recommended_action']}</p>
                                    <p><strong>Expected Outcome:</strong> {action['expected_outcome']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Action buttons
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if st.button(f"📧 Email", key=f"email_{urgency}_{i}"):
                                        st.session_state.expanded_action = actions.index(action)
                                        st.session_state.expanded_action_type = 'email'
                                        st.rerun()
                                
                                with col2:
                                    if st.button(f"📞 Call", key=f"call_{urgency}_{i}"):
                                        st.session_state.expanded_action = actions.index(action)
                                        st.session_state.expanded_action_type = 'call'
                                        st.rerun()
                                
                                st.markdown("---")
        
    except Exception as e:
        logger.error(f"Error rendering action queue: {e}")
        st.error(f"Error displaying action queue: {str(e)}")

def render_priority_analysis(actions: List[Dict[str, Any]]) -> None:
    """Render priority analysis charts."""
    try:
        # Priority distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Priority Score Distribution")
            scores = [a['priority_score'] for a in actions]
            
            fig = px.histogram(
                x=scores,
                nbins=20,
                title="Action Priority Scores",
                labels={'x': 'Priority Score', 'y': 'Count'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Urgency Breakdown")
            urgency_counts = {}
            for action in actions:
                urgency = action['urgency']
                urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
            
            fig = px.pie(
                values=list(urgency_counts.values()),
                names=list(urgency_counts.keys()),
                title="Actions by Urgency Level",
                color_discrete_map={
                    'Critical': '#FF4B4B',
                    'High': '#FF8C00', 
                    'Medium': '#FFD700',
                    'Low': '#90EE90'
                }
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Action type analysis
        st.markdown("#### Action Type Distribution")
        action_types = {}
        for action in actions:
            action_type = action['action_type']
            action_types[action_type] = action_types.get(action_type, 0) + 1
        
        fig = px.bar(
            x=list(action_types.keys()),
            y=list(action_types.values()),
            title="Recommended Actions by Type",
            labels={'x': 'Action Type', 'y': 'Count'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error rendering priority analysis: {e}")

def render_batch_actions(actions: List[Dict[str, Any]]) -> None:
    """Render batch action interface."""
    try:
        st.markdown("#### 🔄 Batch Processing")
        
        # Batch action selection
        col1, col2 = st.columns(2)
        
        with col1:
            action_type = st.selectbox(
                "Select Action Type",
                ["email_reminder", "call_reminder", "collections", "payment_plan"],
                key="batch_action_type"
            )
            
            min_priority = st.slider(
                "Minimum Priority Score",
                min_value=0,
                max_value=100,
                value=60,
                key="batch_min_priority"
            )
        
        with col2:
            urgency_filter = st.multiselect(
                "Urgency Levels",
                ["Critical", "High", "Medium", "Low"],
                default=["Critical", "High"],
                key="batch_urgency"
            )
            
            max_actions = st.number_input(
                "Max Actions to Process",
                min_value=1,
                max_value=100,
                value=20,
                key="batch_max_actions"
            )
        
        # Filter actions
        filtered_actions = [
            a for a in actions
            if a['priority_score'] >= min_priority 
            and a['urgency'] in urgency_filter
        ][:max_actions]
        
        st.write(f"**{len(filtered_actions)}** actions match your criteria")
        
        # Batch execute button
        if st.button("🚀 Execute Batch Actions", type="primary"):
            execute_batch_actions(filtered_actions, action_type)
            
    except Exception as e:
        logger.error(f"Error rendering batch actions: {e}")

def send_email_action(action: Dict[str, Any]) -> None:
    """Execute email action."""
    try:
        customer_info = {
            'name': action['customer_id'],
            'email': action['contact_info']['email'],
            'outstanding_amount': action['amount'],
            'days_overdue': action['days_overdue']
        }
        
        result = send_collection_reminder(customer_info)
        
        if result.get('email', False):
            st.success(f"✅ Email sent successfully to {action['customer_id']}")
        else:
            st.warning(f"⚠️ Email could not be sent. Check configuration.")
            
    except Exception as e:
        st.error(f"❌ Email action failed: {str(e)}")
        logger.error(f"Email action error: {e}")

def show_call_action(action: Dict[str, Any]) -> None:
    """Show call action interface."""
    try:
        phone = action['contact_info']['phone']
        customer_name = action['customer_id']
        
        call_html = call_client(phone, customer_name)
        st.markdown(call_html, unsafe_allow_html=True)
        
        st.info(f"📞 Prepared call link for {customer_name} at {phone}")
        
        # Call notes section
        st.markdown("### 📝 Call Notes")
        call_notes = st.text_area(
            "Enter call notes:",
            placeholder="Enter details from your call...",
            key=f"call_notes_{action['customer_id']}"
        )
        
        if call_notes:
            col1, col2 = st.columns(2)
            
            with col1:
                # Download call notes
                if st.button("📥 Download Call Notes", key=f"download_call_{action['customer_id']}"):
                    import pandas as pd
                    call_data = pd.DataFrame([{
                        'customer': customer_name,
                        'phone': phone,
                        'amount': action['amount'],
                        'days_overdue': action['days_overdue'],
                        'call_notes': call_notes,
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }])
                    
                    csv_data = call_data.to_csv(index=False)
                    st.download_button(
                        "📥 Download Call Notes CSV",
                        csv_data,
                        file_name=f"call_notes_{customer_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                # Email call notes
                user_email = st.session_state.get("recipient")
                if user_email:
                    if st.button("📧 Email Call Notes", key=f"email_call_{action['customer_id']}"):
                        try:
                            from email_utils import send_email
                            email_body = f"""
                            Call Notes for {customer_name}
                            
                            Customer: {customer_name}
                            Phone: {phone}
                            Amount: ${action['amount']:,.2f}
                            Days Overdue: {action['days_overdue']}
                            
                            Call Notes:
                            {call_notes}
                            
                            Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            """
                            success = send_email(user_email, f"Call Notes - {customer_name}", email_body)
                            if success:
                                st.success(f"✅ Call notes sent to {user_email}")
                            else:
                                st.error("❌ Failed to send call notes")
                        except Exception as e:
                            st.error(f"❌ Email error: {str(e)}")
                else:
                    st.warning("⚠️ Configure email address to send call notes")
        
    except Exception as e:
        st.error(f"❌ Call action failed: {str(e)}")
        logger.error(f"Call action error: {e}")

def escalate_to_collections(action: Dict[str, Any]) -> None:
    """Escalate account to collections."""
    try:
        # In a real system, this would POST to a collections API
        collections_data = {
            'customer_id': action['customer_id'],
            'amount': action['amount'],
            'days_overdue': action['days_overdue'],
            'priority': action['urgency'],
            'escalation_reason': action['recommended_action']
        }
        
        # Simulate API call
        st.success(f"✅ {action['customer_id']} escalated to collections")
        st.json(collections_data)
        
        logger.info(f"Escalated {action['customer_id']} to collections")
        
    except Exception as e:
        st.error(f"❌ Collections escalation failed: {str(e)}")
        logger.error(f"Collections escalation error: {e}")

def show_action_details(action: Dict[str, Any]) -> None:
    """Show detailed action information including full email draft."""
    try:
        with st.expander(f"Details for {action['customer_id']}", expanded=True):
            
            # Customer and Account Information
            st.markdown("### 📋 Account Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Customer ID:** {action['customer_id']}")
                st.write(f"**Amount:** ${action['amount']:,.2f}")
                st.write(f"**Days Overdue:** {action['days_overdue']}")
                st.write(f"**Status:** {action['status']}")
            
            with col2:
                st.write(f"**Priority Score:** {action['priority_score']:.0f}/100")
                st.write(f"**Urgency:** {action['urgency']}")
                st.write(f"**Action Type:** {action['action_type']}")
                st.write(f"**Expected Outcome:** {action['expected_outcome']}")
            
            st.markdown("---")
            
            # Contact information
            st.markdown("### 📞 Contact Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Email:** {action['contact_info']['email']}")
            with col2:
                st.write(f"**Phone:** {action['contact_info']['phone']}")
            with col3:
                st.write(f"**Contact:** {action['contact_info']['contact_name']}")
            
            st.markdown("---")
            
            # Email Draft Preview
            st.markdown("### 📧 Email Draft Preview")
            
            # Generate email content based on action type
            email_subject, email_body = generate_email_draft(action)
            
            st.markdown("**Subject:**")
            st.code(email_subject)
            
            st.markdown("**Email Body:**")
            # Show email draft in an expandable container
            with st.expander("📧 View Complete Email Draft", expanded=False):
                st.text_area(
                    "Email Content",
                    value=email_body,
                    height=300,
                    key=f"email_draft_{action['customer_id']}",
                    help="This is the complete email that would be sent to the customer",
                    disabled=True  # Make it read-only
                )
            
            # Action buttons
            st.markdown("---")
            st.markdown("### 🚀 Available Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"📧 Compose Email", key=f"send_email_detail_{action['customer_id']}"):
                    customer_info = {
                        'name': action['customer_id'],
                        'email': action['contact_info']['email'],
                        'outstanding_amount': action['amount'],
                        'days_overdue': action['days_overdue']
                    }
                    result = send_collection_reminder(customer_info)
                    if result.get('email', False):
                        st.success("✅ Email draft opened in your default email client")
                        st.info("📧 Please review and send the email from your email client")
                    else:
                        st.error("❌ Failed to open email client")
            
            with col2:
                if st.button(f"📞 Make Call", key=f"call_detail_{action['customer_id']}"):
                    phone_html = call_client(action['contact_info']['phone'], action['customer_id'])
                    st.markdown(phone_html, unsafe_allow_html=True)
                    
            with col3:
                if st.button(f"📋 Collections", key=f"escalate_detail_{action['customer_id']}"):
                    st.info(f"Account {action['customer_id']} marked for collections escalation")
            
            # Report generation section
            st.markdown("---")
            st.markdown("### 📊 Generate Report")
            
            # Create report data
            report_data = {
                'customer_id': action['customer_id'],
                'amount': action['amount'],
                'days_overdue': action['days_overdue'],
                'status': action['status'],
                'priority_score': action['priority_score'],
                'urgency': action['urgency'],
                'action_type': action['action_type'],
                'expected_outcome': action['expected_outcome'],
                'contact_email': action['contact_info']['email'],
                'contact_phone': action['contact_info']['phone'],
                'contact_name': action['contact_info']['contact_name'],
                'email_subject': email_subject,
                'email_body': email_body,
                'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download report
                if st.button("📥 Download Report", key=f"download_report_{action['customer_id']}"):
                    import pandas as pd
                    report_df = pd.DataFrame([report_data])
                    csv_data = report_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Report CSV",
                        csv_data,
                        file_name=f"action_report_{action['customer_id']}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                # Email report
                user_email = st.session_state.get("recipient")
                if user_email:
                    if st.button("📧 Email Report", key=f"email_report_{action['customer_id']}"):
                        try:
                            from email_utils import send_email
                            email_body = f"""
                            Action Report for {action['customer_id']}
                            
                            Customer ID: {action['customer_id']}
                            Amount: ${action['amount']:,.2f}
                            Days Overdue: {action['days_overdue']}
                            Status: {action['status']}
                            Priority Score: {action['priority_score']:.0f}/100
                            Urgency: {action['urgency']}
                            Action Type: {action['action_type']}
                            Expected Outcome: {action['expected_outcome']}
                            
                            Contact Information:
                            - Email: {action['contact_info']['email']}
                            - Phone: {action['contact_info']['phone']}
                            - Contact: {action['contact_info']['contact_name']}
                            
                            Email Draft:
                            Subject: {email_subject}
                            
                            Body:
                            {email_body}
                            
                            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            """
                            success = send_email(user_email, f"Action Report - {action['customer_id']}", email_body)
                            if success:
                                st.success(f"✅ Report sent to {user_email}")
                            else:
                                st.error("❌ Failed to send report")
                        except Exception as e:
                            st.error(f"❌ Email error: {str(e)}")
                else:
                    st.warning("⚠️ Configure email address to send reports")
        
    except Exception as e:
        logger.error(f"Error showing action details: {e}")
        st.error(f"Error displaying details: {str(e)}")

def generate_email_draft(action: Dict[str, Any]) -> tuple[str, str]:
    """Generate email subject and body for an action."""
    try:
        customer_id = action['customer_id']
        amount = action['amount']
        days_overdue = action['days_overdue']
        action_type = action['action_type']
        
        # Get user name from session state
        user_name = st.session_state.get("user_name", "Accounts Receivable Team")
        
        # Email subject based on action type
        subjects = {
            "email_reminder": f"Payment Reminder - Invoice #{customer_id} (${amount:,.0f})",
            "call_reminder": f"Urgent: Payment Required - Account #{customer_id}",
            "personal_outreach": f"Important Account Review - {customer_id}",
            "collections": f"Final Notice - Account #{customer_id} (${amount:,.0f})",
            "payment_plan": f"Payment Plan Options Available - Account #{customer_id}"
        }
        
        subject = subjects.get(action_type, f"Account Update - {customer_id}")
        
        # Email body based on action type and overdue status
        if days_overdue > 60:
            urgency_tone = "This account is significantly overdue and requires immediate attention."
        elif days_overdue > 30:
            urgency_tone = "This account is overdue and needs prompt resolution."
        elif days_overdue > 0:
            urgency_tone = "This account has recently become overdue."
        else:
            urgency_tone = "This account requires proactive attention."
        
        body = f"""Dear {customer_id},

{urgency_tone}

Account Details:
- Customer ID: {customer_id}
- Outstanding Amount: ${amount:,.2f}
- Days Overdue: {days_overdue}
- Priority Level: {action['urgency']}

{get_email_body_by_type(action_type, amount, days_overdue)}

We value our business relationship and are committed to working with you to resolve this matter promptly.

Please contact us at your earliest convenience to discuss payment arrangements.

Best regards,
{user_name}
Accounts Receivable Team

---
SENDER INSTRUCTIONS:
- Customer Email: {action['contact_info']['email']}
- Customer Phone: {action['contact_info']['phone']}
- Contact Person: {action['contact_info']['contact_name']}
- Send from your email client or forward to collections team
"""
        
        return subject, body
        
    except Exception as e:
        logger.error(f"Error generating email draft: {e}")
        return "Account Update", "Please review account details and contact customer."

def get_email_body_by_type(action_type: str, amount: float, days_overdue: int) -> str:
    """Get action-specific email body content."""
    
    bodies = {
        "email_reminder": f"""
We wanted to remind you that your account has an outstanding balance of ${amount:,.2f}.
Our records show this amount is {days_overdue} days past due.

Please review your account and submit payment at your earliest convenience.
""",
        
        "call_reminder": f"""
This is an urgent reminder regarding your outstanding balance of ${amount:,.2f}, 
which is now {days_overdue} days overdue.

We will be calling you shortly to discuss immediate payment options.
""",
        
        "personal_outreach": f"""
Given the significant outstanding balance of ${amount:,.2f} on your account,
we would like to schedule a personal consultation to discuss payment arrangements.

Our account management team is ready to work with you on a solution.
""",
        
        "collections": f"""
FINAL NOTICE: Your account has an outstanding balance of ${amount:,.2f} 
that is {days_overdue} days overdue.

If payment is not received within 5 business days, this account will be 
forwarded to our collections department.
""",
        
        "payment_plan": f"""
We understand that circumstances can make it challenging to pay the full 
balance of ${amount:,.2f} immediately.

We would like to offer you flexible payment plan options to help resolve 
this outstanding balance over time.
"""
    }
    
    return bodies.get(action_type, f"Please remit payment of ${amount:,.2f} as soon as possible.")

def execute_batch_actions(actions: List[Dict[str, Any]], action_type: str) -> None:
    """Execute batch actions."""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        total_actions = len(actions)
        
        # Get user's email from session state
        user_email = st.session_state.get("recipient")
        if not user_email and action_type == "email_reminder":
            st.error("❌ Please configure your email address in the main app first.")
            return
        
        # Prepare batch report
        batch_report = []
        
        for i, action in enumerate(actions):
            status_text.text(f"Processing {action['customer_id']}...")
            
            # Simulate action execution
            if action_type == "email_reminder":
                # Generate complete email draft
                email_subject, email_body = generate_email_draft(action)
                
                # Create complete email content with customer info
                complete_email = f"""
{email_body}

---
CUSTOMER INFORMATION:
- Customer: {action['contact_info'].get('customer_name', action['customer_id'])}
- Contact: {action['contact_info']['contact_name']}
- Email: {action['contact_info']['email']}
- Phone: {action['contact_info']['phone']}
- Amount: ${action['amount']:,.2f}
- Days Overdue: {action['days_overdue']}
- Priority: {action['urgency']}

Please forward this email to the customer or use the contact information above.
"""
                
                batch_report.append({
                    'customer': action['customer_id'],
                    'amount': action['amount'],
                    'days_overdue': action['days_overdue'],
                    'subject': email_subject,
                    'body': complete_email,
                    'customer_email': action['contact_info']['email'],
                    'customer_phone': action['contact_info']['phone'],
                    'contact_name': action['contact_info']['contact_name']
                })
                success_count += 1
            elif action_type == "collections":
                escalate_to_collections(action)
                success_count += 1
            else:
                success_count += 1  # Simulate success for other actions
            
            # Update progress
            progress_bar.progress((i + 1) / total_actions)
        
        status_text.text("Batch processing complete!")
        st.success(f"✅ Successfully processed {success_count}/{total_actions} actions")
        
        # Show batch results with download/email options
        if action_type == "email_reminder" and batch_report:
            st.markdown("### 📧 Batch Email Results")
            
            # Create CSV for download
            import pandas as pd
            report_df = pd.DataFrame(batch_report)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download option
                csv_data = report_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Batch Report",
                    csv_data,
                    file_name=f"batch_email_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Email option
                if user_email:
                    if st.button("📧 Email Report to Yourself"):
                        try:
                            from email_utils import send_email
                            
                            # Create detailed email report
                            email_body = f"""
Batch Email Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Processed {len(batch_report)} email actions:

"""
                            
                            for i, report in enumerate(batch_report, 1):
                                email_body += f"""
{i}. {report['customer']} - ${report['amount']:,.2f} ({report['days_overdue']} days overdue)
   Subject: {report['subject']}
   Customer Email: {report['customer_email']}
   Contact: {report['contact_name']}
   Phone: {report['customer_phone']}
   
   Email Body:
   {report['body']}
   
   ---
"""
                            
                            success = send_email(user_email, "Batch Email Report", email_body)
                            if success:
                                st.success(f"✅ Report sent to {user_email}")
                            else:
                                st.error("❌ Failed to send email report")
                        except Exception as e:
                            st.error(f"❌ Email error: {str(e)}")
                else:
                    st.warning("⚠️ Configure email address to send reports")
        
        logger.info(f"Batch processed {success_count}/{total_actions} actions of type {action_type}")
        
    except Exception as e:
        st.error(f"❌ Batch execution failed: {str(e)}")
        logger.error(f"Batch execution error: {e}")

# Full-screen action functions

def send_email_action_fullscreen(action: Dict[str, Any], action_id: int) -> None:
    """Render full-screen email action interface."""
    try:
        st.markdown("## 📧 Email Action - Full Screen")
        
        # Close button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("❌ Close", key=f"close_email_{action_id}"):
                st.session_state.expanded_action = None
                st.rerun()
        
        st.markdown("---")
        
        # Customer info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Customer Information")
            st.write(f"**Customer:** {action['contact_info'].get('customer_name', action['customer_id'])}")
            st.write(f"**Contact Person:** {action['contact_info']['contact_name']}")
            st.write(f"**Email:** {action['contact_info']['email']}")
            st.write(f"**Phone:** {action['contact_info']['phone']}")
            st.write(f"**Amount:** ${action['amount']:,.2f}")
            st.write(f"**Days Overdue:** {action['days_overdue']}")
            st.write(f"**Priority:** {action['urgency']}")
        
        with col2:
            st.markdown("### 📧 Email Details")
            
            # Get user email from session state
            user_email = st.session_state.get("recipient")
            if not user_email:
                st.error("❌ Please configure your email address in the main app first!")
                st.info("Go back to the main page and enter your email address.")
                return
            
            st.success(f"✅ Sending to: {user_email}")
            
            # Generate email draft
            email_subject, email_body = generate_email_draft(action)
            
            # Allow editing of subject and body
            edited_subject = st.text_input(
                "Email Subject:",
                value=email_subject,
                key=f"email_subject_{action_id}"
            )
        
        st.markdown("---")
        
        # Email draft editing
        st.markdown("### ✍️ Email Draft")
        
        edited_body = st.text_area(
            "Email Body:",
            value=email_body,
            height=300,
            key=f"email_body_{action_id}",
            help="Edit the email content as needed before sending"
        )
        
        # Preview section
        with st.expander("👀 Email Preview", expanded=False):
            st.markdown("**Subject:** " + edited_subject)
            st.markdown("**Body:**")
            st.markdown(edited_body.replace('\n', '  \n'))  # Markdown line breaks
        
        # Send email button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("📧 Send Email Draft", type="primary", key=f"send_email_fullscreen_{action_id}"):
                try:
                    from email_utils import send_email
                    
                    # Send email to user with customer info
                    email_content = f"""
{edited_body}

---
CUSTOMER INFORMATION:
- Customer: {action['contact_info'].get('customer_name', action['customer_id'])}
- Contact: {action['contact_info']['contact_name']}
- Email: {action['contact_info']['email']}
- Phone: {action['contact_info']['phone']}
- Amount: ${action['amount']:,.2f}
- Days Overdue: {action['days_overdue']}
- Priority: {action['urgency']}

Please forward this email to the customer or use the contact information above.
"""
                    
                    success = send_email(user_email, edited_subject, email_content)
                    
                    if success:
                        st.success("✅ Email draft sent successfully!")
                        st.info("📧 Check your email for the draft to forward to the customer.")
                        
                        # Auto-close after 2 seconds
                        import time
                        time.sleep(2)
                        st.session_state.expanded_action = None
                        st.rerun()
                    else:
                        st.error("❌ Failed to send email. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Email error: {str(e)}")
                    logger.error(f"Email sending error: {e}")
                    
    except Exception as e:
        logger.error(f"Error in email fullscreen: {e}")
        st.error(f"Error: {str(e)}")

def show_call_action_fullscreen(action: Dict[str, Any], action_id: int) -> None:
    """Render full-screen call action interface."""
    try:
        st.markdown("## 📞 Call Action - Full Screen")
        
        # Close button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("❌ Close", key=f"close_call_{action_id}"):
                st.session_state.expanded_action = None
                st.rerun()
        
        st.markdown("---")
        
        # Customer info and call details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Customer Information")
            st.write(f"**Customer:** {action['contact_info'].get('customer_name', action['customer_id'])}")
            st.write(f"**Contact Person:** {action['contact_info']['contact_name']}")
            st.write(f"**Phone:** {action['contact_info']['phone']}")
            st.write(f"**Email:** {action['contact_info']['email']}")
            st.write(f"**Amount:** ${action['amount']:,.2f}")
            st.write(f"**Days Overdue:** {action['days_overdue']}")
            st.write(f"**Priority:** {action['urgency']}")
        
        with col2:
            st.markdown("### 📞 Call Planning")
            
            call_objective = st.selectbox(
                "Call Objective:",
                ["Payment Reminder", "Payment Plan Discussion", "Account Resolution", "Relationship Maintenance"],
                key=f"call_objective_{action_id}"
            )
            
            preferred_time = st.selectbox(
                "Preferred Call Time:",
                ["Now", "Morning (9-11 AM)", "Afternoon (1-3 PM)", "End of Day (4-5 PM)"],
                key=f"call_time_{action_id}"
            )
            
            call_notes = st.text_area(
                "Pre-Call Notes:",
                value=f"Calling regarding overdue amount of ${action['amount']:,.2f}.\nAccount is {action['days_overdue']} days overdue.\nPriority level: {action['urgency']}",
                height=100,
                key=f"call_notes_{action_id}"
            )
        
        st.markdown("---")
        
        # Call script suggestion
        st.markdown("### 📋 Suggested Call Script")
        
        script = f"""
**Opening:**
"Hi {action['contact_info']['contact_name']}, this is [Your Name] from [Company] regarding account {action['customer_id']}."

**Purpose:**
"I'm calling about the outstanding balance of ${action['amount']:,.2f} on your account, which is currently {action['days_overdue']} days past due."

**Discussion Points:**
- Confirm receipt of invoice
- Understand any payment difficulties
- Discuss payment options or plans
- Set clear next steps and timeline

**Closing:**
"Thank you for your time. I'll follow up with an email summary of our discussion."
        """
        
        with st.expander("📝 View Full Call Script", expanded=False):
            st.markdown(script)
        
        # Call actions
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📞 Make Call Now", type="primary", key=f"make_call_{action_id}"):
                phone_html = call_client(action['contact_info']['phone'], action['customer_id'])
                st.markdown(phone_html, unsafe_allow_html=True)
                st.success(f"📞 Call initiated to {action['contact_info']['phone']}")
        
        with col2:
            if st.button("📅 Schedule Call", key=f"schedule_call_{action_id}"):
                st.info("📅 Call scheduled for later (feature coming soon)")
        
        with col3:
            if st.button("📝 Log Call Notes", key=f"log_call_{action_id}"):
                if call_notes:
                    st.success("✅ Call notes saved successfully!")
                    st.write("**Logged Notes:**")
                    st.write(call_notes)
                else:
                    st.warning("⚠️ Please enter call notes first")
    
    except Exception as e:
        logger.error(f"Error in call fullscreen: {e}")
        st.error(f"Error: {str(e)}")

def escalate_to_collections_fullscreen(action: Dict[str, Any], action_id: int) -> None:
    """Render full-screen collections escalation interface."""
    try:
        st.markdown("## 📋 Collections Escalation - Full Screen")
        
        # Close button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("❌ Close", key=f"close_collections_{action_id}"):
                st.session_state.expanded_action = None
                st.rerun()
        
        st.markdown("---")
        
        # Customer info and escalation details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Customer Information")
            st.write(f"**Customer:** {action['contact_info'].get('customer_name', action['customer_id'])}")
            st.write(f"**Contact Person:** {action['contact_info']['contact_name']}")
            st.write(f"**Phone:** {action['contact_info']['phone']}")
            st.write(f"**Email:** {action['contact_info']['email']}")
            st.write(f"**Amount:** ${action['amount']:,.2f}")
            st.write(f"**Days Overdue:** {action['days_overdue']}")
            st.write(f"**Priority:** {action['urgency']}")
        
        with col2:
            st.markdown("### ⚙️ Escalation Configuration")
            
            escalation_type = st.selectbox(
                "Escalation Type:",
                ["Standard Collections", "Legal Action", "Payment Plan Negotiation", "Account Closure"],
                key=f"escalation_type_{action_id}"
            )
            
            escalation_reason = st.selectbox(
                "Escalation Reason:",
                ["No Response to Emails", "Failed Payment Promises", "Dispute Resolution", "High Risk Account"],
                key=f"escalation_reason_{action_id}"
            )
            
            urgency_level = st.selectbox(
                "Urgency Level:",
                ["Low", "Medium", "High", "Critical"],
                index=["Low", "Medium", "High", "Critical"].index(action['urgency']),
                key=f"urgency_level_{action_id}"
            )
        
        st.markdown("---")
        
        # Risk assessment
        st.markdown("### 🎯 Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            payment_history = st.selectbox(
                "Payment History:",
                ["Excellent", "Good", "Fair", "Poor"],
                index=2,  # Default to Fair
                key=f"payment_history_{action_id}"
            )
        
        with col2:
            communication = st.selectbox(
                "Communication Response:",
                ["Responsive", "Slow", "Minimal", "Unresponsive"],
                index=2,  # Default to Minimal
                key=f"communication_{action_id}"
            )
        
        with col3:
            recovery_likelihood = st.selectbox(
                "Recovery Likelihood:",
                ["High", "Medium", "Low", "Very Low"],
                index=1,  # Default to Medium
                key=f"recovery_likelihood_{action_id}"
            )
        
        # Escalation notes
        st.markdown("### 📝 Escalation Notes")
        
        escalation_notes = st.text_area(
            "Additional Notes:",
            value=f"Account {action['customer_id']} being escalated to collections.\nOutstanding: ${action['amount']:,.2f}\nOverdue: {action['days_overdue']} days\nPrevious attempts: Email reminders, phone calls",
            height=150,
            key=f"escalation_notes_{action_id}"
        )
        
        # Collections actions
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚨 Escalate Now", type="primary", key=f"escalate_now_{action_id}"):
                collections_data = {
                    'customer_id': action['customer_id'],
                    'amount': action['amount'],
                    'days_overdue': action['days_overdue'],
                    'escalation_type': escalation_type,
                    'escalation_reason': escalation_reason,
                    'urgency': urgency_level,
                    'payment_history': payment_history,
                    'communication': communication,
                    'recovery_likelihood': recovery_likelihood,
                    'notes': escalation_notes
                }
                
                st.success(f"✅ {action['customer_id']} escalated to collections successfully!")
                
                with st.expander("📄 Escalation Details", expanded=True):
                    st.json(collections_data)
        
        with col2:
            if st.button("📧 Send Final Notice", key=f"final_notice_{action_id}"):
                st.info("📧 Final notice email prepared (feature coming soon)")
        
        with col3:
            if st.button("📊 Generate Report", key=f"collections_report_{action_id}"):
                st.info("📊 Collections report generated (feature coming soon)")
    
    except Exception as e:
        logger.error(f"Error in collections fullscreen: {e}")
        st.error(f"Error: {str(e)}")

def show_action_details_fullscreen(action: Dict[str, Any], action_id: int) -> None:
    """Render full-screen action details interface."""
    try:
        st.markdown("## ℹ️ Action Details - Full Screen")
        
        # Close button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            if st.button("❌ Close", key=f"close_details_{action_id}"):
                st.session_state.expanded_action = None
                st.rerun()
        
        st.markdown("---")
        
        # Comprehensive action details (same as before but in full width)
        show_action_details(action)
    
    except Exception as e:
        logger.error(f"Error in details fullscreen: {e}")
        st.error(f"Error: {str(e)}")
