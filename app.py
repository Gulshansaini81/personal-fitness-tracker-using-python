import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
import datetime
from datetime import timedelta
import json

warnings.filterwarnings('ignore')

# Performance optimization settings
st.set_page_config(
    page_title="Personal Fitness Tracker",
    layout="wide",
    initial_sidebar_state="collapsed"  # Reduces initial render time
)

# Cache frequently used functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_datasets(data_dir):
    """Cache the initial loading of datasets"""
    datasets = {}
    
    # Try to load calories.csv (Check both data directory and root directory)
    calories_found = False
    try:
        # First try in data directory
        calories_path = os.path.join(data_dir, "calories.csv")
        if os.path.exists(calories_path):
            datasets['calories'] = pd.read_csv(calories_path)
            calories_found = True
        else:
            # Then try in root directory
            root_calories_path = "calories.csv"
            if os.path.exists(root_calories_path):
                datasets['calories'] = pd.read_csv(root_calories_path)
                calories_found = True
                
        if not calories_found:
            st.warning(f"calories.csv not found. Looked in: {calories_path} and {os.path.abspath(root_calories_path)}")
            datasets['calories'] = None
            
    except Exception as e:
        st.error(f"Error loading calories.csv: {str(e)}")
        datasets['calories'] = None
        
    # Try to load exercise.csv (Check both data directory and root directory)
    exercise_found = False
    try:
        # First try in data directory
        exercise_path = os.path.join(data_dir, "exercise.csv")
        if os.path.exists(exercise_path):
            datasets['exercise'] = pd.read_csv(exercise_path)
            exercise_found = True
        else:
            # Then try in root directory
            root_exercise_path = "exercise.csv"
            if os.path.exists(root_exercise_path):
                datasets['exercise'] = pd.read_csv(root_exercise_path)
                exercise_found = True
                
        if not exercise_found:
            st.warning(f"exercise.csv not found. Looked in: {exercise_path} and {os.path.abspath(root_exercise_path)}")
            datasets['exercise'] = None
            
    except Exception as e:
        st.error(f"Error loading exercise.csv: {str(e)}")
        datasets['exercise'] = None
        
    return datasets

@st.cache_resource  # Cache the model in memory
def train_prediction_model(X_train, y_train):
    """Train and cache the prediction model"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Data Management Class
class FitnessDataManager:
    """Centralized data manager for the fitness tracker application"""
    
    def __init__(self):
        """Initialize the data manager and load existing data"""
        self.data_dir = "data"
        self.ensure_data_directory()
        
        # Initialize fitness data storage using lazy loading
        if 'fitness_data' not in st.session_state:
            # Only load existing data, don't create sample data
            st.session_state.fitness_data = {
                'user_history': self.load_user_history(),
                'workout_plan': self.load_workout_plan(),
                'weekly_goals': self.load_weekly_goals(),
                'last_updated': datetime.datetime.now().isoformat()
            }
            
            # Set initialization flag to prevent automatic data addition
            st.session_state.first_load = True
            st.session_state.prevent_auto_updates = True
            
        # Cache the datasets
        self.cached_datasets = load_datasets(self.data_dir)
    
    def ensure_data_directory(self):
        """Ensure the data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def load_user_history(self):
        """Load user fitness history from file"""
        history_file = os.path.join(self.data_dir, "user_history.csv")
        
        try:
            if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
                # Use optimized pandas reading with only necessary columns and types
                history = pd.read_csv(
                    history_file,
                    parse_dates=["Date"],  # Pre-parse dates for efficiency
                    infer_datetime_format=True
                )
                # Convert date to just the date part once for efficiency
                if "Date" in history.columns:
                    history["Date"] = history["Date"].dt.date
                return history
            else:
                # Return empty DataFrame with expected columns instead of sample data
                return pd.DataFrame(columns=[
                    "Date", "Age", "BMI", "Duration", "Heart_Rate", 
                    "Body_Temp", "Calories_Burned", "Source"
                ])
        except:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "Date", "Age", "BMI", "Duration", "Heart_Rate", 
                "Body_Temp", "Calories_Burned", "Source"
            ])
    
    def load_workout_plan(self):
        """Load workout plan from file"""
        workout_file = os.path.join(self.data_dir, "workout_plan.csv")
        try:
            plan = pd.read_csv(workout_file)
            if "Date" in plan.columns:
                plan["Date"] = pd.to_datetime(plan["Date"]).dt.date
            return plan
        except:
            return pd.DataFrame(columns=[
                "Date", "Type", "Exercise", "Duration", "Calories", "Completed"
            ])
    
    def load_weekly_goals(self):
        """Load weekly goals from file"""
        goals_file = os.path.join(self.data_dir, "goals.json")
        try:
            with open(goals_file, 'r') as f:
                return json.load(f)
        except:
            # Default goals
            return {
                'weekly_calorie_goal': 3000,
                'weekly_workout_goal': 4
            }
    
    def add_history_entry(self, entry_data):
        """Add a new entry to user history"""
        # Temporarily disable prevent_auto_updates to allow explicit additions
        st.session_state.explicit_save_requested = True
        
        # Convert to DataFrame if not already
        if not isinstance(entry_data, pd.DataFrame):
            entry_data = pd.DataFrame([entry_data])
        
        # Add source if not present
        if 'Source' not in entry_data.columns:
            entry_data['Source'] = 'prediction'
        
        # Efficiently add to session state using concat with optimized parameters
        if len(st.session_state.fitness_data['user_history']) > 0:
            st.session_state.fitness_data['user_history'] = pd.concat(
                [st.session_state.fitness_data['user_history'], entry_data],
                ignore_index=True,  # More efficient than reset_index(drop=True)
                copy=False  # Avoid unnecessary data copying
            )
        else:
            st.session_state.fitness_data['user_history'] = entry_data
        
        # Save to file
        self.save_user_history()
        
        # Update last updated timestamp
        st.session_state.fitness_data['last_updated'] = datetime.datetime.now().isoformat()
        
        # Reset flag
        st.session_state.explicit_save_requested = False
    
    def save_user_history(self):
        """Save user history to file"""
        # Check if we should prevent auto-updates
        if st.session_state.get('prevent_auto_updates', False) and not st.session_state.get('explicit_save_requested', False):
            return
            
        history_file = os.path.join(self.data_dir, "user_history.csv")
        
        # Only save if data exists and has changed
        if 'user_history' in st.session_state.fitness_data and len(st.session_state.fitness_data['user_history']) > 0:
            # Convert dates to string format before saving
            history = st.session_state.fitness_data['user_history'].copy()
            
            # Use optimized CSV writing
            history.to_csv(history_file, index=False, date_format='%Y-%m-%d')
    
    def save_workout_plan(self):
        """Save workout plan to file"""
        # Check if we should prevent auto-updates
        if st.session_state.get('prevent_auto_updates', False) and not st.session_state.get('explicit_save_requested', False):
            return
            
        workout_file = os.path.join(self.data_dir, "workout_plan.csv")
        st.session_state.fitness_data['workout_plan'].to_csv(workout_file, index=False)
    
    def save_weekly_goals(self):
        """Save weekly goals to file"""
        # Check if we should prevent auto-updates
        if st.session_state.get('prevent_auto_updates', False) and not st.session_state.get('explicit_save_requested', False):
            return
            
        goals_file = os.path.join(self.data_dir, "goals.json")
        with open(goals_file, 'w') as f:
            json.dump(st.session_state.fitness_data['weekly_goals'], f)
    
    def add_workout_entry(self, workout_data, add_to_history=False):
        """Add a new workout to plan and optionally to history"""
        # Create a copy of the workout data
        workout = workout_data.copy()
        
        # Convert date to string if it's a datetime object
        if isinstance(workout["Date"], (datetime.date, datetime.datetime)):
            workout["Date"] = workout["Date"].strftime("%Y-%m-%d")
        
        # Add to workout plan
        workout_df = pd.DataFrame([workout])
        
        if len(st.session_state.fitness_data['workout_plan']) > 0:
            st.session_state.fitness_data['workout_plan'] = pd.concat([
                st.session_state.fitness_data['workout_plan'], workout_df
            ]).reset_index(drop=True)
        else:
            st.session_state.fitness_data['workout_plan'] = workout_df
            
        # Save the updated workout plan
        self.save_workout_plan()
        
        # Optionally add to history (only if explicitly requested)
        if add_to_history:
            # Create history entry from workout
            history_entry = {
                'Date': workout["Date"],
                'Age': st.session_state.get('user_age', None),  
                'BMI': st.session_state.get('user_bmi', None),  
                'Duration': workout["Duration"],
                'Heart_Rate': 120,  # Estimated heart rate during workout
                'Body_Temp': 38,    # Estimated body temp during workout
                'Calories_Burned': workout["Calories"],
                'Source': 'workout'
            }
            
            # Add to history (but don't call this function recursively)
            self.add_history_entry(history_entry)
    
    def update_workout_status(self, workout_index, completed=True):
        """Mark a workout as completed or not completed"""
        try:
            st.session_state.fitness_data['workout_plan'].at[workout_index, 'Completed'] = completed
            self.save_workout_plan()
            return True
        except:
            return False
    
    def update_weekly_goals(self, goal_data):
        """Update weekly goals"""
        st.session_state.fitness_data['weekly_goals'].update(goal_data)
        self.save_weekly_goals()
        return True
    
    # Get statistics for the current week with caching
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_weekly_stats_cached(_self, user_history, workout_plan, weekly_goals):
        """Cached version of weekly stats calculation"""
        today = datetime.datetime.now().date()
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        
        # Initialize stats
        stats = {
            'total_calories': 0,
            'calorie_goal_progress': 0,
            'workouts_completed': 0,
            'workout_goal_progress': 0
        }
        
        # Calculate total calories burned this week
        if len(user_history) > 0:
            # Convert Date to datetime if it's not already
            if isinstance(user_history["Date"].iloc[0], str):
                user_history_dates = pd.to_datetime(user_history["Date"]).dt.date
            else:
                user_history_dates = user_history["Date"]
                
            this_week = user_history[
                (user_history_dates >= start_of_week) & 
                (user_history_dates <= end_of_week)
            ]
            
            if len(this_week) > 0:
                stats['total_calories'] = this_week["Calories_Burned"].sum()
        
        # Count completed workouts this week
        if len(workout_plan) > 0:
            # Convert Date to datetime if it's not already
            if isinstance(workout_plan["Date"].iloc[0], str):
                workout_plan_dates = pd.to_datetime(workout_plan["Date"]).dt.date
            else:
                workout_plan_dates = workout_plan["Date"]
                
            this_week_workouts = workout_plan[
                (workout_plan_dates >= start_of_week) & 
                (workout_plan_dates <= end_of_week) &
                (workout_plan["Completed"] == True)
            ]
            
            stats['workouts_completed'] = len(this_week_workouts)
        
        # Calculate progress percentages
        weekly_calorie_goal = weekly_goals.get('weekly_calorie_goal', 3000)
        weekly_workout_goal = weekly_goals.get('weekly_workout_goal', 4)
        
        stats['calorie_goal_progress'] = min(stats['total_calories'] / weekly_calorie_goal, 1.0) if weekly_calorie_goal > 0 else 0
        stats['workout_goal_progress'] = min(stats['workouts_completed'] / weekly_workout_goal, 1.0) if weekly_workout_goal > 0 else 0
        
        return stats
    
    def get_weekly_stats(self):
        """Get statistics for the current week"""
        return self.get_weekly_stats_cached(
            st.session_state.fitness_data['user_history'],
            st.session_state.fitness_data['workout_plan'],
            st.session_state.fitness_data['weekly_goals']
        )

# Initialize the data manager - only once per session
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = FitnessDataManager()

data_manager = st.session_state.data_manager

st.write("# Personal Fitness Tracker")
st.write("Track fitness, plan workouts, and achieve your health goals!")

# Create tab structure
tabs = st.tabs(["Dashboard", "Visualizations", "Workout Planner"])

with tabs[0]:  # Dashboard Tab
    st.write("## 📊 Fitness Dashboard")
    st.write("This dashboard helps you predict calories burned during exercise and track your fitness goals.")
    
    # Only show informational message about sample data on very first load
    if st.session_state.get('first_load', False):
        st.info("👋 Welcome! Enter your details below to start tracking your fitness. No sample data has been added.")
        # Reset flag after showing welcome message
        st.session_state.first_load = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Stats")
        
        # Use None for initial values to require user input
        age = st.number_input("Age", min_value=15, max_value=80, value=None, placeholder="Enter your age")
        gender = st.radio("Gender", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", min_value=40, max_value=160, value=None, placeholder="Enter your weight")
        height = st.number_input("Height (cm)", min_value=140, max_value=210, value=None, placeholder="Enter your height")
        
        # Calculate BMI and store values only if all inputs are provided
        if age is not None and weight is not None and height is not None:
            # Calculate BMI
            bmi = weight / ((height/100) ** 2)
            
            # Only store values when explicitly requested
            if st.button("Save Stats to Session"):
                st.session_state['user_age'] = age
                st.session_state['user_gender'] = gender
                st.session_state['user_weight'] = weight
                st.session_state['user_height'] = height
                st.session_state['user_bmi'] = bmi
                st.success("Stats saved to current session")
            
            st.write(f"### BMI: **{bmi:.1f}**")
            
            # BMI Classification
            def classify_bmi(bmi):
                if bmi < 18.5:
                    return "Underweight", "⚠️ You may need to gain some weight!"
                elif 18.5 <= bmi < 24.9:
                    return "Normal Weight", "✅ You have a healthy weight!"
                elif 25 <= bmi < 29.9:
                    return "Overweight", "⚠️ You may need to watch your diet and exercise more!"
                else:
                    return "Obese", "❗ It's important to maintain a healthier lifestyle!"

            bmi_category, advice = classify_bmi(bmi)
            st.write(f"### BMI Category: **{bmi_category}**")
            st.info(advice)
        else:
            st.warning("Please enter all required data (age, weight, height)")
            
        # Exercise inputs with default values but ensure they don't cause automatic prediction
        st.subheader("Exercise Details")
        duration = st.slider("Duration (minutes)", min_value=10, max_value=120, value=30)
        heart_rate = st.slider("Heart Rate (bpm)", min_value=60, max_value=200, value=130)
        body_temp = st.slider("Body Temperature (°C)", min_value=36.0, max_value=41.0, value=37.0, step=0.1)
        
        # Track if user has interacted with sliders
        if 'slider_values_initialized' not in st.session_state:
            st.session_state.slider_values_initialized = True
            st.session_state.previous_duration = duration
            st.session_state.previous_heart_rate = heart_rate
            st.session_state.previous_body_temp = body_temp
            st.session_state.user_adjusted_sliders = False
        
        # Check if user adjusted any slider
        if (duration != st.session_state.previous_duration or 
            heart_rate != st.session_state.previous_heart_rate or 
            body_temp != st.session_state.previous_body_temp):
            st.session_state.user_adjusted_sliders = True
        
        # Update previous values
        st.session_state.previous_duration = duration
        st.session_state.previous_heart_rate = heart_rate
        st.session_state.previous_body_temp = body_temp
        
        # Only enable prediction button if all values are provided AND user has either explicitly adjusted sliders or provided personal stats
        user_provided_personal_stats = (age is not None and weight is not None and height is not None)
        predict_disabled = not (user_provided_personal_stats and (st.session_state.user_adjusted_sliders or user_provided_personal_stats))
        predict_clicked = st.button("Predict Calories Burned", disabled=predict_disabled)
    
    with col2:
        try:
            # Load and preprocess data only when needed
            calories = data_manager.cached_datasets.get('calories')
            exercise = data_manager.cached_datasets.get('exercise')
            
            if calories is None or exercise is None:
                st.error("Error: Required dataset files not found. Please ensure calories.csv and exercise.csv are in the data directory.")
            else:
                # Merge datasets
                df = pd.merge(exercise, calories, on='User_ID')
                
                # Select features and target
                X = df[['Age', 'Gender', 'Weight', 'Height', 'Duration', 'Heart_Rate', 'Body_Temp']]
                y = df['Calories']
                
                # Encode Gender: Male=1, Female=0
                X['Gender'] = X['Gender'].map({'male': 1, 'female': 0})
                
                # Split the data for training
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Use cached model or train a new one
                model = train_prediction_model(X_train, y_train)
                
                if predict_clicked and age is not None and weight is not None and height is not None and duration is not None and heart_rate is not None and body_temp is not None:
                    # Use the model to make predictions
                    input_data = [{
                        'Age': age,
                        'Gender': 1 if gender == "Male" else 0,
                        'Weight': weight,
                        'Height': height,
                        'Duration': duration,
                        'Heart_Rate': heart_rate,
                        'Body_Temp': body_temp
                    }]
                    
                    # Convert to DataFrame
                    input_df = pd.DataFrame(input_data)
                    
                    # Make prediction
                    predicted_calories = model.predict(input_df)[0]
                    
                    st.success(f"## Estimated Calories Burned: **{predicted_calories:.2f}** kcal")
                    
                    # Option to save prediction - EXPLICIT BUTTON
                    if st.button("💾 Save This Prediction to Database"):
                        current_date = datetime.datetime.now().date()
                        
                        # Create new history entry with user-provided values only
                        new_entry = {
                            'Date': current_date,
                            'Age': age,
                            'BMI': weight / ((height/100) ** 2),
                            'Duration': duration,
                            'Heart_Rate': heart_rate,
                            'Body_Temp': body_temp,
                            'Calories_Burned': predicted_calories,
                            'Source': 'prediction'
                        }
                        
                        # Explicitly set save request flag
                        st.session_state.explicit_save_requested = True
                        
                        # Add to history using data manager
                        data_manager.add_history_entry(new_entry)
                        
                        # Reset flag
                        st.session_state.explicit_save_requested = False
                        
                        st.success("✅ Prediction saved to your history database!")
                elif predict_clicked:
                    st.error("Please fill in all required fields before predicting.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

with tabs[1]:  # Visualizations Tab
    st.write("## 📈 Fitness Visualizations")
    st.write("Analyze your fitness data with interactive charts and visualizations.")
    
    # Check if there's data available
    user_history = st.session_state.fitness_data['user_history']
    
    if len(user_history) == 0:
        st.warning("⚠️ You don't have any workout data yet. Start tracking your workouts to see visualizations!")
    else:
        # Convert Date to datetime if it's a string
        if isinstance(user_history["Date"].iloc[0], str):
            user_history["Date"] = pd.to_datetime(user_history["Date"])
            
        # Create visualization tabs
        viz_tabs = st.tabs(["Calories Over Time", "Workout Duration", "Weekly Summary"])
        
        with viz_tabs[0]:  # Calories Over Time
            st.subheader("Calories Burned Over Time")
            
            # Sort by date
            sorted_history = user_history.sort_values("Date")
            
            # Create a chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sorted_history["Date"], sorted_history["Calories_Burned"], marker='o', linestyle='-')
            ax.set_xlabel("Date")
            ax.set_ylabel("Calories Burned")
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Display the chart
            st.pyplot(fig)
            
            # Display statistics
            st.write("### Calorie Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Calories", f"{user_history['Calories_Burned'].mean():.1f}")
            with col2:
                st.metric("Maximum Calories", f"{user_history['Calories_Burned'].max():.1f}")
            with col3:
                st.metric("Total Calories", f"{user_history['Calories_Burned'].sum():.1f}")
        
        with viz_tabs[1]:  # Workout Duration
            st.subheader("Workout Duration Analysis")
            
            # Create a bar chart for duration
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(sorted_history["Date"], sorted_history["Duration"], color='teal')
            ax.set_xlabel("Date")
            ax.set_ylabel("Duration (minutes)")
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Display the chart
            st.pyplot(fig)
            
            # Display statistics
            st.write("### Duration Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Duration", f"{user_history['Duration'].mean():.1f} mins")
            with col2:
                st.metric("Maximum Duration", f"{user_history['Duration'].max():.1f} mins")
            with col3:
                st.metric("Total Duration", f"{user_history['Duration'].sum():.1f} mins")
        
        with viz_tabs[2]:  # Weekly Summary
            st.subheader("Weekly Fitness Summary")
            
            # Get current week's data
            today = datetime.datetime.now().date()
            start_of_week = today - timedelta(days=today.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            
            # Convert Date to date object if it's a datetime
            if isinstance(user_history["Date"].iloc[0], datetime.datetime):
                date_column = user_history["Date"].dt.date
            else:
                date_column = user_history["Date"]
                
            # Filter for this week
            this_week = user_history[
                (date_column >= start_of_week) & 
                (date_column <= end_of_week)
            ]
            
            # Weekly stats
            weekly_stats = data_manager.get_weekly_stats()
            
            # Create progress bars
            st.write("### Weekly Goals Progress")
            st.progress(weekly_stats['calorie_goal_progress'], text=f"Calorie Goal: {weekly_stats['total_calories']:.1f} / {st.session_state.fitness_data['weekly_goals']['weekly_calorie_goal']} kcal")
            st.progress(weekly_stats['workout_goal_progress'], text=f"Workout Goal: {weekly_stats['workouts_completed']} / {st.session_state.fitness_data['weekly_goals']['weekly_workout_goal']} workouts")
            
            # Scatter plot of calories vs duration for this week
            if len(this_week) > 0:
                st.write("### Calories vs Duration This Week")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns_plot = sn.scatterplot(x="Duration", y="Calories_Burned", hue="Source", data=this_week, s=100, ax=ax)
                ax.set_xlabel("Duration (minutes)")
                ax.set_ylabel("Calories Burned")
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No workouts recorded for this week yet.")
                
            # Update weekly goals
            st.write("### Update Weekly Goals")
            col1, col2 = st.columns(2)
            with col1:
                new_calorie_goal = st.number_input(
                    "Weekly Calorie Goal", 
                    value=st.session_state.fitness_data['weekly_goals']['weekly_calorie_goal'],
                    min_value=500,
                    max_value=10000,
                    step=100
                )
            with col2:
                new_workout_goal = st.number_input(
                    "Weekly Workout Goal", 
                    value=st.session_state.fitness_data['weekly_goals']['weekly_workout_goal'],
                    min_value=1,
                    max_value=14,
                    step=1
                )
            
            if st.button("Update Goals"):
                data_manager.update_weekly_goals({
                    'weekly_calorie_goal': new_calorie_goal,
                    'weekly_workout_goal': new_workout_goal
                })
                st.success("Weekly goals updated successfully!")

with tabs[2]:  # Workout Planner Tab
    st.write("## 🏋️ Workout Planner")
    st.write("Create and manage your personalized workout plan.")
    
    # Workout Plan Form
    st.subheader("Add New Workout")
    
    with st.form("workout_form"):
        # Form for creating a new workout
        col1, col2 = st.columns(2)
        
        with col1:
            workout_date = st.date_input("Workout Date", datetime.datetime.now().date())
            workout_type = st.selectbox("Workout Type", [
                "Cardio", "Strength Training", "HIIT", "Yoga", "Pilates", 
                "Swimming", "Cycling", "Running", "Walking", "Other"
            ])
            
        with col2:
            workout_exercise = st.text_input("Exercise Name", placeholder="e.g., Push-ups, Treadmill, etc.")
            workout_duration = st.number_input("Duration (minutes)", min_value=5, max_value=180, value=30)
            estimated_calories = st.number_input("Estimated Calories", min_value=50, max_value=2000, value=300)
        
        # Add to history option
        add_to_history = st.checkbox("Also add to workout history", value=True)
        
        # Submit button
        submitted = st.form_submit_button("Add to Workout Plan")
        
        if submitted:
            if workout_exercise == "":
                st.error("Please enter an exercise name.")
            else:
                # Create workout data
                workout_data = {
                    "Date": workout_date,
                    "Type": workout_type,
                    "Exercise": workout_exercise,
                    "Duration": workout_duration,
                    "Calories": estimated_calories,
                    "Completed": False
                }
                
                # Add to workout plan
                data_manager.add_workout_entry(workout_data, add_to_history=add_to_history)
                
                st.success(f"✅ {workout_type} workout added to your plan for {workout_date}!")
    
    # Display workout plan
    st.subheader("Your Workout Plan")
    
    workout_plan = st.session_state.fitness_data['workout_plan']
    
    if len(workout_plan) == 0:
        st.info("You don't have any workouts planned. Use the form above to add workouts to your plan.")
    else:
        # Ensure Date column is properly parsed before sorting
        try:
            # Make sure we have data to work with
            if len(workout_plan) == 0:
                workout_dates = []
                sorted_workouts = workout_plan
            else:
                # Attempt to convert dates explicitly with proper error handling
                try:
                    # Convert to datetime if it's not already
                    if not pd.api.types.is_datetime64_any_dtype(workout_plan["Date"]):
                        workout_plan = workout_plan.copy()  # Make a copy to avoid modifying original
                        workout_plan["Date"] = pd.to_datetime(workout_plan["Date"], errors='coerce')
                        
                        # Drop rows with invalid dates
                        workout_plan = workout_plan.dropna(subset=['Date'])
                    
                    # Now we can safely sort the workouts by date
                    sorted_workouts = workout_plan.sort_values("Date").reset_index(drop=True)
                    
                    # Get unique dates for grouping, but only if we have datetime data
                    if len(sorted_workouts) > 0 and pd.api.types.is_datetime64_any_dtype(sorted_workouts["Date"]):
                        workout_dates = sorted_workouts["Date"].dt.date.unique()
                    else:
                        workout_dates = []
                        
                except Exception as sort_error:
                    st.error(f"Error processing workout dates: {sort_error}")
                    sorted_workouts = workout_plan
                    workout_dates = []
        except Exception as e:
            st.error(f"Unexpected error in workout planner: {e}")
            sorted_workouts = workout_plan
            workout_dates = []
        
        # For each date, show workouts
        for date in workout_dates:
            # Format the date as a string for display
            try:
                # Format date for display
                if hasattr(date, 'strftime'):
                    date_str = date.strftime("%A, %b %d, %Y")
                else:
                    date_str = str(date)
                
                # Get workouts for this date
                # We need to convert both the date from workout_dates and the dates in the DataFrame 
                # to the same format for comparison
                if pd.api.types.is_datetime64_any_dtype(sorted_workouts["Date"]):
                    # Convert both sides to date objects for comparison
                    if isinstance(date, datetime.datetime):
                        date_for_comparison = date.date()
                    else:
                        date_for_comparison = date
                        
                    # Filter using date (not datetime)
                    day_workouts = sorted_workouts[sorted_workouts["Date"].dt.date == date_for_comparison]
                else:
                    # If we don't have proper datetime objects, try string matching
                    day_workouts = sorted_workouts[sorted_workouts["Date"].astype(str) == str(date)]
                
                # Check if we should expand this date by default
                is_today = False
                today = datetime.datetime.now().date()
                try:
                    # Handle different date types for comparison
                    if hasattr(date, 'date') and callable(date.date):
                        date_for_comparison = date.date()
                    elif isinstance(date, datetime.date):
                        date_for_comparison = date
                    else:
                        # Try to parse the date string
                        try:
                            date_for_comparison = pd.to_datetime(date).date()
                        except:
                            date_for_comparison = None
                    
                    # Now compare with today's date
                    if date_for_comparison is not None:
                        is_today = (date_for_comparison == today)
                except Exception:
                    # If anything goes wrong, just don't auto-expand
                    is_today = False
                
                with st.expander(f"📅 {date_str} ({len(day_workouts)} workouts)", expanded=is_today):
                    for idx, workout in day_workouts.iterrows():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        
                        with col1:
                            st.write(f"**{workout['Exercise']}** ({workout['Type']})")
                            st.write(f"Duration: {workout['Duration']} mins | Calories: {workout['Calories']} kcal")
                        
                        with col2:
                            status = "✅ Completed" if workout["Completed"] else "⏳ Pending"
                            st.write(status)
                        
                        with col3:
                            # Toggle completion status
                            current_status = workout["Completed"]
                            new_status = st.checkbox(
                                "Done", 
                                value=current_status, 
                                key=f"workout_{idx}",
                                help="Mark as completed"
                            )
                            
                            # Update status if changed
                            if new_status != current_status:
                                data_manager.update_workout_status(idx, completed=new_status)
                                st.experimental_rerun()
                        
                        st.divider()
        
        # Summary of completed workouts
        total_workouts = len(workout_plan)
        completed_workouts = len(workout_plan[workout_plan["Completed"] == True])
        completion_rate = (completed_workouts / total_workouts) * 100 if total_workouts > 0 else 0
        
        st.write("### Workout Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Workouts", total_workouts)
        with col2:
            st.metric("Completed", completed_workouts)
        with col3:
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
            
        # Clear all completed workouts button
        if completed_workouts > 0 and st.button("Clear Completed Workouts"):
            # Remove completed workouts
            st.session_state.fitness_data['workout_plan'] = workout_plan[workout_plan["Completed"] == False].reset_index(drop=True)
            data_manager.save_workout_plan()
            st.success(f"Removed {completed_workouts} completed workouts!")
            st.experimental_rerun()
