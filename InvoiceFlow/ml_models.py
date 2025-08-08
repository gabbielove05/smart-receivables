"""
Machine Learning Models for Smart Receivables Navigator
Implements priority scoring, anomaly detection, and model explainability with SHAP.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import streamlit as st

# ML libraries
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Try to import SHAP for model explainability
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP library available for model explainability")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP library not available - model explanations will be simplified")

class ReceivablesMLModel:
    """Main ML model class for receivables analytics."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models."""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Amount-based features
            if 'amount' in df.columns:
                features['amount'] = df['amount'].fillna(0)
                features['amount_log'] = np.log1p(features['amount'])
                features['amount_zscore'] = (features['amount'] - features['amount'].mean()) / features['amount'].std()
            
            # Customer-based features
            if 'customer_id' in df.columns:
                customer_stats = df.groupby('customer_id').agg({
                    'amount': ['sum', 'count', 'mean', 'std']
                }).fillna(0)
                customer_stats.columns = ['customer_total', 'customer_count', 'customer_avg', 'customer_std']
                
                # Map back to original dataframe
                for col in customer_stats.columns:
                    features[col] = df['customer_id'].map(customer_stats[col]).fillna(0)
                
                # Customer risk indicators
                features['is_high_volume_customer'] = (features['customer_count'] > features['customer_count'].quantile(0.8)).astype(int)
                features['is_high_value_customer'] = (features['customer_total'] > features['customer_total'].quantile(0.8)).astype(int)
            
            # Date-based features
            current_date = datetime.now()
            
            if 'issue_date' in df.columns:
                try:
                    issue_dates = pd.to_datetime(df['issue_date'], errors='coerce')
                    features['days_since_issue'] = (current_date - issue_dates).dt.days.fillna(30)
                    features['issue_day_of_week'] = issue_dates.dt.dayofweek.fillna(1)
                    features['issue_month'] = issue_dates.dt.month.fillna(6)
                except:
                    features['days_since_issue'] = 30
                    features['issue_day_of_week'] = 1
                    features['issue_month'] = 6
            else:
                features['days_since_issue'] = np.random.randint(1, 90, len(df))
                features['issue_day_of_week'] = np.random.randint(0, 7, len(df))
                features['issue_month'] = np.random.randint(1, 13, len(df))
            
            if 'due_date' in df.columns:
                try:
                    due_dates = pd.to_datetime(df['due_date'], errors='coerce')
                    features['days_overdue'] = np.maximum(0, (current_date - due_dates).dt.days.fillna(0))
                    features['payment_terms'] = (due_dates - pd.to_datetime(df['issue_date'], errors='coerce')).dt.days.fillna(30)
                except:
                    features['days_overdue'] = np.maximum(0, features['days_since_issue'] - 30)
                    features['payment_terms'] = 30
            else:
                features['days_overdue'] = np.maximum(0, features['days_since_issue'] - 30)
                features['payment_terms'] = 30
            
            # Status-based features
            if 'status' in df.columns:
                status_encoder = LabelEncoder()
                features['status_encoded'] = status_encoder.fit_transform(df['status'].fillna('unknown'))
                
                # Binary status features
                features['is_paid'] = (df['status'].str.lower() == 'paid').astype(int)
                features['is_overdue'] = (df['status'].str.lower() == 'overdue').astype(int)
                features['is_outstanding'] = (df['status'].str.lower() == 'outstanding').astype(int)
            else:
                # Infer status from days overdue
                features['status_encoded'] = np.where(features['days_overdue'] > 0, 2, 1)  # 2=overdue, 1=outstanding, 0=paid
                features['is_paid'] = (features['days_overdue'] == 0).astype(int)
                features['is_overdue'] = (features['days_overdue'] > 30).astype(int)
                features['is_outstanding'] = ((features['days_overdue'] > 0) & (features['days_overdue'] <= 30)).astype(int)
            
            # Risk score features
            features['amount_risk'] = np.where(features.get('amount', 0) > features.get('amount', 0).quantile(0.9), 2,
                                              np.where(features.get('amount', 0) > features.get('amount', 0).quantile(0.7), 1, 0))
            
            features['age_risk'] = np.where(features['days_overdue'] > 90, 3,
                                           np.where(features['days_overdue'] > 30, 2,
                                                   np.where(features['days_overdue'] > 0, 1, 0)))
            
            # Seasonal features
            features['is_quarter_end'] = (features['issue_month'] % 3 == 0).astype(int)
            features['is_year_end'] = (features['issue_month'] == 12).astype(int)
            
            # Fill any remaining NaN values
            features = features.fillna(0)
            
            self.feature_names = list(features.columns)
            logger.info(f"Prepared {len(features.columns)} features for ML model")
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            # Return minimal feature set
            return pd.DataFrame({
                'amount': df.get('amount', [0] * len(df)),
                'days_overdue': [0] * len(df)
            })
    
    def train_anomaly_detector(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Train anomaly detection model using Isolation Forest."""
        try:
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_estimators=100,
                max_samples='auto'
            )
            
            iso_forest.fit(features_scaled)
            
            # Get anomaly scores
            anomaly_scores = iso_forest.decision_function(features_scaled)
            anomaly_labels = iso_forest.predict(features_scaled)
            
            # Store model and scaler
            self.models['anomaly_detector'] = iso_forest
            self.scalers['anomaly_scaler'] = scaler
            
            logger.info("Anomaly detection model trained successfully")
            
            return {
                'model': iso_forest,
                'scaler': scaler,
                'scores': anomaly_scores,
                'labels': anomaly_labels,
                'n_anomalies': (anomaly_labels == -1).sum()
            }
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            return {'error': str(e)}
    
    def train_priority_classifier(self, features: pd.DataFrame, target: pd.Series = None) -> Dict[str, Any]:
        """Train priority classification model."""
        try:
            # Create target variable if not provided
            if target is None:
                target = self.create_priority_target(features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Gradient Boosting classifier
            gb_classifier = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            gb_classifier.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = gb_classifier.score(X_train_scaled, y_train)
            test_score = gb_classifier.score(X_test_scaled, y_test)
            
            # Get predictions and probabilities
            y_pred = gb_classifier.predict(X_test_scaled)
            y_pred_proba = gb_classifier.predict_proba(X_test_scaled)
            
            # Store model and scaler
            self.models['priority_classifier'] = gb_classifier
            self.scalers['priority_scaler'] = scaler
            
            logger.info(f"Priority classifier trained: Train Score={train_score:.3f}, Test Score={test_score:.3f}")
            
            return {
                'model': gb_classifier,
                'scaler': scaler,
                'train_score': train_score,
                'test_score': test_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'feature_importance': gb_classifier.feature_importances_
            }
            
        except Exception as e:
            logger.error(f"Error training priority classifier: {e}")
            return {'error': str(e)}
    
    def create_priority_target(self, features: pd.DataFrame) -> pd.Series:
        """Create priority target variable based on business rules."""
        try:
            priority_scores = np.zeros(len(features))
            
            # Amount-based priority
            if 'amount' in features.columns:
                amount_percentiles = features['amount'].quantile([0.5, 0.8, 0.95])
                priority_scores += np.where(features['amount'] > amount_percentiles[0.95], 3,
                                          np.where(features['amount'] > amount_percentiles[0.8], 2,
                                                  np.where(features['amount'] > amount_percentiles[0.5], 1, 0)))
            
            # Days overdue priority
            if 'days_overdue' in features.columns:
                priority_scores += np.where(features['days_overdue'] > 90, 3,
                                          np.where(features['days_overdue'] > 30, 2,
                                                  np.where(features['days_overdue'] > 0, 1, 0)))
            
            # Customer volume priority
            if 'customer_total' in features.columns:
                customer_percentiles = features['customer_total'].quantile([0.7, 0.9])
                priority_scores += np.where(features['customer_total'] > customer_percentiles[0.9], 2,
                                          np.where(features['customer_total'] > customer_percentiles[0.7], 1, 0))
            
            # Convert to categorical priority levels
            priority_labels = np.where(priority_scores >= 6, 'Critical',
                                     np.where(priority_scores >= 4, 'High',
                                             np.where(priority_scores >= 2, 'Medium', 'Low')))
            
            return pd.Series(priority_labels, index=features.index)
            
        except Exception as e:
            logger.error(f"Error creating priority target: {e}")
            return pd.Series(['Low'] * len(features), index=features.index)
    
    def get_model_explanations(self, features: pd.DataFrame, model_type: str = 'priority_classifier') -> Dict[str, Any]:
        """Get SHAP explanations for model predictions."""
        try:
            if not SHAP_AVAILABLE:
                return self.get_simple_explanations(features, model_type)
            
            if model_type not in self.models:
                logger.warning(f"Model {model_type} not found")
                return {'error': f'Model {model_type} not trained'}
            
            model = self.models[model_type]
            scaler = self.scalers.get(f"{model_type.split('_')[0]}_scaler")
            
            if scaler is None:
                logger.warning(f"Scaler for {model_type} not found")
                return {'error': 'Model scaler not found'}
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Create SHAP explainer
            if isinstance(model, GradientBoostingClassifier):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(features_scaled)
            else:
                # For other models, use KernelExplainer (slower but more general)
                sample_size = min(100, len(features_scaled))
                background = features_scaled[:sample_size]
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(features_scaled[:sample_size])
            
            return {
                'explainer': explainer,
                'shap_values': shap_values,
                'feature_names': self.feature_names,
                'expected_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting model explanations: {e}")
            return self.get_simple_explanations(features, model_type)
    
    def get_simple_explanations(self, features: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Get simplified feature importance when SHAP is not available."""
        try:
            if model_type not in self.models:
                return {'error': f'Model {model_type} not trained'}
            
            model = self.models[model_type]
            
            # Get feature importance from tree-based models
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = dict(zip(self.feature_names, importances))
                
                # Sort by importance
                sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                
                return {
                    'feature_importance': sorted_importance,
                    'top_features': list(sorted_importance.keys())[:10],
                    'explanation_type': 'feature_importance'
                }
            else:
                return {'error': 'Model does not support feature importance'}
                
        except Exception as e:
            logger.error(f"Error getting simple explanations: {e}")
            return {'error': str(e)}

def train_priority_model(df: pd.DataFrame) -> Dict[str, Any]:
    """Train priority model for receivables analysis."""
    try:
        logger.info("Training priority model")
        
        if df.empty:
            return {'error': 'Empty dataset provided'}
        
        # Initialize ML model
        ml_model = ReceivablesMLModel()
        
        # Prepare features
        features = ml_model.prepare_features(df)
        
        # Train priority classifier
        priority_results = ml_model.train_priority_classifier(features)
        
        # Train anomaly detector
        anomaly_results = ml_model.train_anomaly_detector(features)
        
        return {
            'priority_model': priority_results,
            'anomaly_model': anomaly_results,
            'features': features,
            'ml_instance': ml_model
        }
        
    except Exception as e:
        logger.error(f"Priority model training failed: {e}")
        return {'error': str(e)}

def get_next_best_actions(df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get next best actions using ML models."""
    try:
        # Train models first
        model_results = train_priority_model(df)
        
        if 'error' in model_results:
            logger.warning(f"ML model training failed: {model_results['error']}")
            return get_rule_based_actions(df, config)
        
        ml_instance = model_results.get('ml_instance')
        features = model_results.get('features')
        
        if ml_instance and features is not None:
            # Generate priority scores
            priority_scores = calculate_priority_scores_ml(features, ml_instance)
            
            # Create action recommendations
            actions = create_action_recommendations_ml(df, priority_scores, config)
            
            return sorted(actions, key=lambda x: x.get('priority_score', 0), reverse=True)[:20]
        else:
            return get_rule_based_actions(df, config)
            
    except Exception as e:
        logger.error(f"Error getting next best actions: {e}")
        return get_rule_based_actions(df, config)

def get_rule_based_actions(df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate rule-based actions as fallback."""
    try:
        actions = []
        
        for idx, row in df.iterrows():
            amount = float(row.get('amount', 0)) if pd.notnull(row.get('amount')) else 0
            customer_id = str(row.get('customer_id', f'CUST_{idx+1}'))
            status = str(row.get('status', 'outstanding'))
            
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
                'days_overdue': 0,
                'status': status,
                'action_type': action_type,
                'recommended_action': f"Contact {customer_id} regarding ${amount:,.0f}",
                'urgency': get_urgency_level_ml(priority),
                'expected_outcome': f"Collect ${amount * 0.7:,.0f}",
                'contact_info': generate_contact_info_ml(customer_id)
            }
            
            actions.append(action)
        
        return sorted(actions, key=lambda x: x['priority_score'], reverse=True)[:20]
        
    except Exception as e:
        logger.error(f"Error generating rule-based actions: {e}")
        return []

def calculate_priority_scores_ml(features: pd.DataFrame, ml_instance: ReceivablesMLModel) -> np.ndarray:
    """Calculate priority scores using ML model."""
    try:
        if 'anomaly_detector' in ml_instance.models:
            anomaly_model = ml_instance.models['anomaly_detector']
            scaler = ml_instance.scalers.get('anomaly_scaler')
            
            if scaler is not None:
                features_scaled = scaler.transform(features)
                anomaly_scores = anomaly_model.decision_function(features_scaled)
                
                # Convert to priority scores (higher = higher priority)
                priority_scores = -anomaly_scores
                
                # Normalize to 0-100 scale
                if priority_scores.max() > priority_scores.min():
                    priority_scores = ((priority_scores - priority_scores.min()) / 
                                      (priority_scores.max() - priority_scores.min()) * 100)
                else:
                    priority_scores = np.full(len(features), 50.0)
                
                return priority_scores
        
        # Fallback to simple scoring
        return np.random.uniform(20, 100, len(features))
        
    except Exception as e:
        logger.error(f"Error calculating ML priority scores: {e}")
        return np.random.uniform(20, 100, len(features))

def create_action_recommendations_ml(df: pd.DataFrame, priority_scores: np.ndarray, 
                                    config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create action recommendations using ML priority scores."""
    try:
        actions = []
        
        for idx, score in enumerate(priority_scores):
            if idx >= len(df):
                break
                
            row = df.iloc[idx]
            amount = float(row.get('amount', 0)) if pd.notnull(row.get('amount')) else 0
            customer_id = str(row.get('customer_id', f'CUST_{idx+1}'))
            status = str(row.get('status', 'unknown'))
            
            # Determine action type based on score and data
            if score > 80:
                action_type = "personal_outreach"
            elif score > 60:
                action_type = "call_reminder" 
            elif score > 40:
                action_type = "email_reminder"
            else:
                action_type = "payment_plan"
            
            action = {
                'priority_score': float(score),
                'customer_id': customer_id,
                'amount': amount,
                'days_overdue': 0,
                'status': status,
                'action_type': action_type,
                'recommended_action': f"ML-recommended {action_type.replace('_', ' ')} for {customer_id} - ${amount:,.0f}",
                'urgency': get_urgency_level_ml(score),
                'expected_outcome': f"Expected recovery: ${amount * 0.75:,.0f}",
                'contact_info': generate_contact_info_ml(customer_id)
            }
            
            actions.append(action)
        
        return actions
        
    except Exception as e:
        logger.error(f"Error creating ML action recommendations: {e}")
        return []

def get_urgency_level_ml(priority_score: float) -> str:
    """Convert priority score to urgency level."""
    if priority_score >= 80:
        return "Critical"
    elif priority_score >= 60:
        return "High"
    elif priority_score >= 40:
        return "Medium"
    else:
        return "Low"

def generate_contact_info_ml(customer_id: str) -> Dict[str, str]:
    """Generate consistent contact information."""
    import hashlib
    
    try:
        # Generate consistent fake data based on customer_id
        hash_obj = hashlib.md5(str(customer_id).encode())
        hash_hex = hash_obj.hexdigest()
        
        # Use hash to generate consistent contact info
        phone_suffix = int(hash_hex[:4], 16) % 10000
        email_prefix = hash_hex[:8]
        
        return {
            'email': f"{email_prefix}@{str(customer_id).lower()}.com",
            'phone': f"+1-555-{phone_suffix:04d}",
            'contact_name': f"Contact {str(customer_id)[-3:]}"
        }
    except Exception as e:
        logger.error(f"Error generating contact info: {e}")
        return {
            'email': f"contact@{str(customer_id).lower()}.com",
            'phone': "+1-555-0000",
            'contact_name': f"Contact {str(customer_id)}"
        }

def train_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Train all ML models on the provided dataset."""
    try:
        logger.info("Starting ML model training")
        
        if df.empty:
            logger.warning("Cannot train models on empty dataset")
            return {'error': 'Empty dataset provided'}
        
        # Initialize model
        ml_model = ReceivablesMLModel()
        
        # Prepare features
        features = ml_model.prepare_features(df)
        
        if features.empty:
            logger.warning("No features could be prepared from dataset")
            return {'error': 'Feature preparation failed'}
        
        results = {}
        
        # Train anomaly detector
        anomaly_results = ml_model.train_anomaly_detector(features)
        results['anomaly_detection'] = anomaly_results
        
        # Train priority classifier
        priority_results = ml_model.train_priority_classifier(features)
        results['priority_classification'] = priority_results
        
        # Mark model as trained
        ml_model.is_trained = True
        
        # Store in session state for reuse
        if 'ml_model' not in st.session_state:
            st.session_state.ml_model = ml_model
        
        logger.info("ML model training completed successfully")
        return {
            'success': True,
            'model': ml_model,
            'results': results,
            'features_shape': features.shape,
            'feature_names': ml_model.feature_names
        }
        
    except Exception as e:
        logger.error(f"ML model training failed: {e}")
        return {'error': str(e)}

def get_model_explanations(df: pd.DataFrame, record_index: int = 0) -> Dict[str, Any]:
    """Get model explanations for a specific record."""
    try:
        # Check if model exists in session state
        if 'ml_model' not in st.session_state:
            logger.warning("No trained model found in session state")
            return {'error': 'No trained model available'}
        
        ml_model = st.session_state.ml_model
        
        if not ml_model.is_trained:
            logger.warning("Model is not trained")
            return {'error': 'Model not trained'}
        
        # Prepare features for the specific record
        features = ml_model.prepare_features(df)
        
        if record_index >= len(features):
            logger.warning(f"Record index {record_index} out of range")
            return {'error': 'Record index out of range'}
        
        # Get explanations for the specific record
        record_features = features.iloc[[record_index]]
        explanations = ml_model.get_model_explanations(record_features, 'priority_classifier')
        
        return explanations
        
    except Exception as e:
        logger.error(f"Error getting model explanations: {e}")
        return {'error': str(e)}

def get_next_best_actions(df: pd.DataFrame, top_n: int = 20) -> List[Dict[str, Any]]:
    """Get ML-powered next-best-action recommendations."""
    try:
        # Check if model exists
        if 'ml_model' not in st.session_state:
            # Train model if not available
            train_result = train_models(df)
            if 'error' in train_result:
                logger.warning("Using rule-based actions due to ML training failure")
                return get_rule_based_actions(df, top_n)
        
        ml_model = st.session_state.ml_model
        features = ml_model.prepare_features(df)
        
        # Get anomaly scores
        if 'anomaly_detector' in ml_model.models and 'anomaly_scaler' in ml_model.scalers:
            scaler = ml_model.scalers['anomaly_scaler']
            anomaly_model = ml_model.models['anomaly_detector']
            
            features_scaled = scaler.transform(features)
            anomaly_scores = anomaly_model.decision_function(features_scaled)
            
            # Convert to priority scores (invert anomaly scores)
            priority_scores = 100 * (1 - (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min()))
        else:
            # Fallback to simple priority scoring
            priority_scores = calculate_simple_priority_scores(df, features)
        
        # Create action recommendations
        actions = []
        for i, (idx, row) in enumerate(df.iterrows()):
            if i >= top_n:
                break
            
            action = create_action_from_ml_prediction(row, priority_scores[i], features.iloc[i])
            actions.append(action)
        
        # Sort by priority score
        actions.sort(key=lambda x: x['priority_score'], reverse=True)
        
        logger.info(f"Generated {len(actions)} ML-powered action recommendations")
        return actions
        
    except Exception as e:
        logger.error(f"Error generating ML actions: {e}")
        return get_rule_based_actions(df, top_n)

def calculate_simple_priority_scores(df: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    """Calculate simple priority scores when ML models are not available."""
    try:
        scores = np.zeros(len(df))
        
        # Amount-based scoring
        if 'amount' in features.columns:
            amount_scores = (features['amount'] - features['amount'].min()) / (features['amount'].max() - features['amount'].min())
            scores += amount_scores * 40  # 40% weight
        
        # Days overdue scoring
        if 'days_overdue' in features.columns:
            overdue_scores = np.clip(features['days_overdue'] / 90, 0, 1)  # Cap at 90 days
            scores += overdue_scores * 30  # 30% weight
        
        # Customer value scoring
        if 'customer_total' in features.columns:
            customer_scores = (features['customer_total'] - features['customer_total'].min()) / (features['customer_total'].max() - features['customer_total'].min())
            scores += customer_scores * 20  # 20% weight
        
        # Status-based scoring
        if 'is_overdue' in features.columns:
            scores += features['is_overdue'] * 10  # 10% weight for overdue status
        
        return scores
        
    except Exception as e:
        logger.error(f"Error calculating simple priority scores: {e}")
        return np.random.random(len(df)) * 100

def create_action_from_ml_prediction(row: pd.Series, priority_score: float, features: pd.Series) -> Dict[str, Any]:
    """Create action recommendation from ML model prediction."""
    try:
        customer_id = row.get('customer_id', f'CUST_{row.name}')
        amount = row.get('amount', 0)
        status = row.get('status', 'unknown')
        
        # Determine urgency based on priority score
        if priority_score >= 80:
            urgency = 'Critical'
        elif priority_score >= 60:
            urgency = 'High'
        elif priority_score >= 40:
            urgency = 'Medium'
        else:
            urgency = 'Low'
        
        # Determine action type based on features
        days_overdue = features.get('days_overdue', 0)
        
        if days_overdue > 90 or priority_score >= 90:
            action_type = 'collections'
        elif days_overdue > 30 or urgency == 'Critical':
            action_type = 'call_reminder'
        elif amount > 50000 or urgency == 'High':
            action_type = 'personal_outreach'
        else:
            action_type = 'email_reminder'
        
        return {
            'customer_id': customer_id,
            'amount': amount,
            'priority_score': priority_score,
            'urgency': urgency,
            'action_type': action_type,
            'days_overdue': days_overdue,
            'status': status,
            'ml_confidence': min(1.0, priority_score / 100),
            'recommended_action': get_action_description(action_type, customer_id, amount, days_overdue),
            'expected_outcome': get_expected_outcome(action_type, amount),
            'contact_info': generate_contact_info(customer_id)
        }
        
    except Exception as e:
        logger.error(f"Error creating ML action: {e}")
        return {
            'customer_id': 'UNKNOWN',
            'amount': 0,
            'priority_score': 0,
            'urgency': 'Low',
            'action_type': 'email_reminder',
            'error': str(e)
        }

def get_rule_based_actions(df: pd.DataFrame, top_n: int) -> List[Dict[str, Any]]:
    """Fallback rule-based action generation."""
    try:
        actions = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            if i >= top_n:
                break
            
            customer_id = row.get('customer_id', f'CUST_{idx}')
            amount = row.get('amount', 0)
            status = row.get('status', 'outstanding')
            
            # Simple rule-based priority
            if amount > 100000:
                priority = 90
                urgency = 'Critical'
                action_type = 'personal_outreach'
            elif status == 'overdue':
                priority = 80
                urgency = 'High'
                action_type = 'call_reminder'
            elif amount > 50000:
                priority = 70
                urgency = 'High'
                action_type = 'email_reminder'
            else:
                priority = 50
                urgency = 'Medium'
                action_type = 'email_reminder'
            
            action = {
                'customer_id': customer_id,
                'amount': amount,
                'priority_score': priority,
                'urgency': urgency,
                'action_type': action_type,
                'days_overdue': 0,
                'status': status,
                'recommended_action': get_action_description(action_type, customer_id, amount, 0),
                'expected_outcome': get_expected_outcome(action_type, amount),
                'contact_info': generate_contact_info(customer_id)
            }
            
            actions.append(action)
        
        return sorted(actions, key=lambda x: x['priority_score'], reverse=True)
        
    except Exception as e:
        logger.error(f"Error generating rule-based actions: {e}")
        return []

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
    
    return {
        'email': f"{email_prefix}@{customer_id.lower().replace('_', '')}.com",
        'phone': f"+1-555-{phone_suffix:04d}",
        'contact_name': f"Contact {customer_id[-3:]}"
    }

def render_model_explanation_modal(explanation_data: Dict[str, Any], customer_id: str) -> None:
    """Render model explanation in an expandable section."""
    try:
        with st.expander(f"🧠 Why was {customer_id} flagged? (Model Explanation)", expanded=False):
            if 'error' in explanation_data:
                st.error(f"❌ {explanation_data['error']}")
                return
            
            if explanation_data.get('explanation_type') == 'feature_importance':
                # Simple feature importance explanation
                st.markdown("### 📊 Feature Importance")
                
                feature_importance = explanation_data['feature_importance']
                top_features = list(feature_importance.keys())[:5]
                
                for feature in top_features:
                    importance = feature_importance[feature]
                    st.metric(
                        label=feature.replace('_', ' ').title(),
                        value=f"{importance:.3f}",
                        help=f"Importance score for {feature}"
                    )
            
            elif 'shap_values' in explanation_data:
                # SHAP explanation
                st.markdown("### 🎯 SHAP Analysis")
                st.info("SHAP values show how each feature contributes to the model's prediction")
                
                # Display top contributing features
                if len(explanation_data['shap_values']) > 0:
                    shap_values = explanation_data['shap_values'][0]  # First class for multiclass
                    feature_names = explanation_data['feature_names']
                    
                    # Create feature contribution chart
                    contributions = dict(zip(feature_names, shap_values))
                    sorted_contributions = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True))
                    
                    top_contributions = dict(list(sorted_contributions.items())[:8])
                    
                    fig = go.Figure(go.Bar(
                        x=list(top_contributions.values()),
                        y=list(top_contributions.keys()),
                        orientation='h',
                        marker_color=['red' if x < 0 else 'green' for x in top_contributions.values()]
                    ))
                    
                    fig.update_layout(
                        title="Feature Contributions (SHAP Values)",
                        xaxis_title="SHAP Value",
                        yaxis_title="Features",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # General explanation
            st.markdown("### 💡 Interpretation Guide")
            st.info("""
            **High Priority Factors:**
            • High outstanding amounts
            • Extended overdue periods
            • Customer payment history
            • Seasonal payment patterns
            
            **Model Confidence:** Models are trained on historical patterns and may not capture all business context.
            Always apply business judgment alongside model recommendations.
            """)
            
    except Exception as e:
        logger.error(f"Error rendering model explanation: {e}")
        st.error("❌ Unable to display model explanation")

