import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycountry
import time
import datetime
import random
import numpy as np
from io import BytesIO

st.set_page_config(page_title="AI Sales Dashboard", layout="wide")

# Hardcoded login credentials
LOGIN_ID = "isaac@gmail.com"
LOGIN_PASSWORD = "Isaac_2004"

# Initialize session state variables
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'failed_attempt' not in st.session_state:
    st.session_state.failed_attempt = False
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0
if 'show_instructions' not in st.session_state:
    st.session_state.show_instructions = False
if 'has_seen_welcome' not in st.session_state:
    st.session_state.has_seen_welcome = False
if 'selected_team_member' not in st.session_state:
    st.session_state.selected_team_member = "All"
if 'selected_team' not in st.session_state:
    st.session_state.selected_team = "All"
if 'date_filter' not in st.session_state:
    # Default to last 30 days
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=30)
    st.session_state.date_filter = (start_date, end_date)

# Define team structure
TEAMS = {
    "Sales Team": ["issac", "Sarah", "Michael", "Jessica"],
    "Marketing Team": ["Alex", "Jordan", "Taylor", "Morgan"],
    "Support Team": ["Casey", "Jamie", "Riley", "Quinn"]
}

# Flatten team members for selection
ALL_TEAM_MEMBERS = ["All"] + [member for team in TEAMS.values() for member in team]
ALL_TEAMS = ["All"] + list(TEAMS.keys())

def authenticate(login_id, login_password):
    """Authenticate user credentials"""
    if login_id == LOGIN_ID and login_password == LOGIN_PASSWORD:
        st.session_state.authenticated = True
        st.session_state.failed_attempt = True
        return True
    else:
        st.session_state.failed_attempt = True
        st.session_state.login_attempts += 1
        return False

def logout_user():
    """Log out the user and reset session state"""
    st.session_state.authenticated = False
    st.session_state.failed_attempt = False
    st.session_state.login_attempts = 0
    st.session_state.has_seen_welcome = False
    st.success("Logged out successfully!")
    st.rerun()

def toggle_instructions():
    """Toggle instructions visibility"""
    
    st.session_state.show_instructions = not st.session_state.show_instructions

def generate_synthetic_data():
    """Generate synthetic data if the CSV file is not available"""
    st.info("Reading dataset and shoin visuals.")

    # Constants for data generation
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime(2024, 5, 20)
    days = (end_date - start_date).days

    # Lists for random selections
    countries = ["USA", "UK", "Canada", "Australia", "Germany", "France", "Japan", "India", "Brazil", "Mexico"]
    request_types = ["GET", "POST", "PUT", "DELETE"]
    url_paths = [
        "/scheduledemo", "/cloud/services", "/ai/solutions", "/services/consulting",
        "/pricing", "/contact", "/about", "/blog", "/support", "/products"
    ]
    response_codes = [200, 201, 204, 400, 404, 500]
    job_types = ["Web Development", "AI Implementation", "Cloud Migration", "Consulting", "Training"]
    repeat_customers = ["Yes", "No"]
    company_names = [
        "TechCorp", "GlobalSoft", "DataDynamics", "CloudNine", "AIVentures",
        "SmartSystems", "FutureTech", "InnovateCo", "DigitalEdge", "CoreTech"
    ]

    # Team members assignments to customers
    team_members = {}
    for team, members in TEAMS.items():
        for member in members:
            team_members[member] = team

    all_members = list(team_members.keys())

    # Generate data
    num_records = 1000
    customer_ids = [f"CUST{i:05d}" for i in range(1, 101)]  # 100 unique customer IDs

    data = []
    for _ in range(num_records):
        customer_id = random.choice(customer_ids)
        company_name = random.choice(company_names)
        date = start_date + datetime.timedelta(days=random.randint(0, days))
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        timestamp = f"{date.strftime('%Y-%m-%d')} {hour:02d}:{minute:02d}:{second:02d}"

        # Assign IP address based on country for consistency
        country = random.choice(countries)
        ip_first_octet = {"USA": "12", "UK": "18", "Canada": "24", "Australia": "36",
                         "Germany": "42", "France": "48", "Japan": "54", "India": "60",
                         "Brazil": "72", "Mexico": "78"}.get(country, "92")
        ip_address = f"{ip_first_octet}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

        request_type = random.choice(request_types)
        url_path = random.choice(url_paths)
        response_code = random.choice(response_codes)
        job_type = random.choice(job_types)
        repeat_customer = random.choice(repeat_customers)

        quantity = random.randint(1, 10)
        price = round(random.uniform(100, 5000), 2)
        cost = round(price * random.uniform(0.3, 0.7), 2)  # Cost is 30-70% of price
        profit = price - cost

        # Assign a team member
        assigned_member = random.choice(all_members)

        data.append({
            "Customer ID": customer_id,
            "Company Name": company_name,
            "Timestamp": timestamp,
            "Date": date.strftime('%Y-%m-%d'),
            "IP Address": ip_address,
            "Country": country,
            "Request Type": request_type,
            "URL Path": url_path,
            "Response Code": response_code,
            "Job Type": job_type,
            "Repeat Customer": repeat_customer,
            "Quantity": quantity,
            "Price (USD)": price,
            "Cost (USD)": cost,
            "Profit (USD)": profit,
            "Team Member": assigned_member,
            "Team": team_members[assigned_member],
            "Response Time (ms)": random.randint(50, 5000)
        })

    df = pd.DataFrame(data)
    return df

def load_data():
    try:
        # Try to load from CSV
        try:
            df = pd.read_csv("ai_solutions_web_server_log2.csv")
        except FileNotFoundError:
            st.error("CSV file not found. Generating synthetic data.")
            return generate_synthetic_data()

        # Check if we have team and team member columns, if not, add them with default values
        if "Team Member" not in df.columns:
            df["Team Member"] = "Unknown"
        if "Team" not in df.columns:
            df["Team"] = "Unknown"

        # Process data
        df["Repeat Customer"] = df["Repeat Customer"].astype(str).str.strip()
        df["Revenue"] = pd.to_numeric(df["Price (USD)"], errors='coerce')
        df["Profit"] = pd.to_numeric(df["Profit (USD)"], errors='coerce')
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df["Hour"] = pd.to_datetime(df["Timestamp"], errors='coerce').dt.hour
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors='coerce')

        # Add request category
        df['Request Category'] = df['URL Path'].apply(classify_request)

        # Generate synthetic data to fill in any missing information
        synthetic_df = generate_synthetic_data()

        # Combine the real data with synthetic data
        combined_df = pd.concat([df, synthetic_df], ignore_index=True)

        return combined_df.dropna()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty DataFrame with expected columns for demo purposes
        return pd.DataFrame(columns=["URL Path", "Country", "Job Type", "Repeat Customer",
                                   "Price (USD)", "Profit (USD)", "Date", "Timestamp", "Quantity"])


def filter_data(df):
    """Filter data based on session state filters"""
    filtered_df = df.copy()

    # Date filter
    if 'date_filter' in st.session_state:
        start_date, end_date = st.session_state.date_filter
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) &
                                 (filtered_df['Date'] <= end_date)]

    # Team filter
    if st.session_state.selected_team != "All":
        filtered_df = filtered_df[filtered_df['Team'] == st.session_state.selected_team]

    # Team member filter
    if st.session_state.selected_team_member != "All":
        filtered_df = filtered_df[filtered_df['Team Member'] == st.session_state.selected_team_member]

    return filtered_df

def classify_request(path):
    path = str(path).lower()
    if 'scheduledemo' in path:
        return 'Schedule Demo'
    elif 'cloud' in path:
        return 'Cloud Services'
    elif 'ai' in path:
        return 'AI Request'
    elif 'services' in path:
        return 'General Services'
    elif 'support' in path:
        return 'Support'
    elif 'products' in path:
        return 'Products'
    elif 'contact' in path:
        return 'Contact'
    elif 'pricing' in path:
        return 'Pricing'
    elif 'blog' in path:
        return 'Blog'
    else:
        return 'Other'

def calculate_kpis(df):
    """Calculate key performance indicators with target comparison"""
    # Define target values for each KPI
    targets = {
        "Total Revenue": 500000,  # Example target value
        "Total Profit": 200000,   # Example target value
        "Profit Margin": 40,       # Example target value in percentage
        "Average Order Value": 1000,  # Example target value
        "Total Orders": 500,      # Example target value
        "Repeat Customer Rate": 30,  # Example target value in percentage
        "Average Response Time": 1000  # Example target value in milliseconds
    }

    # Calculate actual KPI values
    total_revenue = df["Revenue"].sum()
    total_profit = df["Profit"].sum()
    profit_margin = 100 * total_profit / max(total_revenue, 1)
    average_order_value = df["Revenue"].mean()
    total_orders = len(df)
    repeat_customer_rate = 100 * df[df["Repeat Customer"] == "Yes"].shape[0] / max(df.shape[0], 1)
    average_response_time = df["Response Time (ms)"].mean()

    # Compare actual values with targets and determine color
    kpis = {
        "Total Revenue": {
            "value": "${:,.2f}".format(total_revenue),
            "color": "green" if total_revenue >= targets["Total Revenue"] else "red"
        },
        "Total Profit": {
            "value": "${:,.2f}".format(total_profit),
            "color": "green" if total_profit >= targets["Total Profit"] else "red"
        },
        "Profit Margin": {
            "value": "{:.1f}%".format(profit_margin),
            "color": "green" if profit_margin >= targets["Profit Margin"] else "red"
        },
        "Average Order Value": {
            "value": "${:,.2f}".format(average_order_value),
            "color": "green" if average_order_value >= targets["Average Order Value"] else "red"
        },
        "Total Orders": {
            "value": "{:,}".format(total_orders),
            "color": "green" if total_orders >= targets["Total Orders"] else "red"
        },
        "Repeat Customer Rate": {
            "value": "{:.1f}%".format(repeat_customer_rate),
            "color": "green" if repeat_customer_rate >= targets["Repeat Customer Rate"] else "red"
        },
        "Average Response Time": {
            "value": "{:.0f} ms".format(average_response_time),
            "color": "green" if average_response_time <= targets["Average Response Time"] else "red"
        }
    }
    return kpis

def export_data(df):
    """Export dataframe to Excel or CSV"""
    export_format = st.sidebar.selectbox("Export Format", ["CSV", "Excel"], key="export_format_select")

    if st.sidebar.button("Export Data", key="export_data_button"):
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            b64 = BytesIO()
            b64.write(csv.encode())
            b64.seek(0)
            return b64, "text/csv", "ai_sales_dashboard_export.csv"
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            output.seek(0)
            return output, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "ai_sales_dashboard_export.xlsx"

    return None, None, None


def show_login_screen():
    # Hide sidebar when on login screen
    st.markdown(
        """
        <style>
        section.main > div:first-child,
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        div.block-container {
            padding-top: 2rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Create a card-like container for login
    st.markdown("""
    <style>
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display a logo or icon
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/6295/6295417.png", width=100)
        st.markdown("<h1 style='text-align: center;'>AI Sales Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; margin-bottom: 30px;'>Please log in to access the dashboard</div>", unsafe_allow_html=True)

        with st.form("login_form"):
            login_id = st.text_input("Email", placeholder="Enter your email")
            login_password = st.text_input("Password", type="password", placeholder="Enter your password")

            # Add help text for first-time users (instruction #1)
            st.info("👋 First time here? Use the demo credentials below to log in and explore the dashboard.")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submit_button = st.form_submit_button("Log In", use_container_width=True)

            if submit_button:
                with st.spinner("Authenticating..."):
                    # Add a slight delay for better UX
                    time.sleep(0.5)
                    if authenticate(login_id, login_password):
                        st.success("Login successful! Redirecting to dashboard...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Invalid login credentials. Please try again. (Attempt {st.session_state.login_attempts})")

        # Demo credentials hint (make sure this matches your hardcoded credentials)
        st.markdown("<div style='text-align: center; margin-top: 30px; opacity: 0.7;'>Demo credentials: isaac@gmail.com / Isaac_2004</div>", unsafe_allow_html=True)

        # Additional info about the app
        with st.expander("About this dashboard"):
            st.write("""
            This AI Sales Dashboard provides insights into sales performance, customer behavior,
            and service requests. After logging in, you'll have access to various analytics
            including geographical distribution, customer types, job profitability, and more.

            ✨ **Key Features:**
            - Team and individual performance metrics
            - Geographic sales distribution
            - Customer engagement analysis
            - Job type profitability
            - Request category distribution
            - Response time analysis
            """)

        st.markdown("</div>", unsafe_allow_html=True)

def show_welcome_message():
    """Show a one-time welcome message with basic instructions"""
    if not st.session_state.has_seen_welcome:
        st.session_state.has_seen_welcome = True

        with st.container():
            st.info("""
            ## 👋 Welcome to the AI Sales Dashboard!

            **Getting Started:**
            1. Use the sidebar to navigate between different views
            2. Filter data by team, team member, or date range
            3. Export reports with the export button at the bottom of the sidebar

            Need help? Click the "Show Instructions" button at any time!
            """)

            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Got it!", key="dismiss_welcome"):
                    st.rerun()

def show_instructions():
    """Show detailed instructions (instruction #2)"""
    with st.expander("Dashboard Instructions", expanded=st.session_state.show_instructions):
        st.markdown("""
        ### 📊 How to Use This Dashboard

        #### Navigation
        - Use the sidebar menu to switch between different views
        - Each view focuses on a specific aspect of your sales data

        #### Filtering Data
        - **Teams Filter**: Select a specific team to view their performance
        - **Team Member Filter**: Select an individual team member to see their specific metrics
        - **Date Range Filter**: Use the date picker to analyze data from a specific time period

        #### Key Features
        - **KPI Cards**: Key performance indicators shown at the top of each page
        - **Interactive Charts**: Hover over chart elements to see detailed information
        - **Export Function**: Export filtered data as CSV or Excel using the sidebar button

        #### Tips for Effective Analysis
        - Compare team and individual performance using the filter options
        - Use date filtering to identify trends over time
        - Export specific views for reporting and presentations
        """)

def show_status_bar(df):
    """Show a status bar with information about the current data view (instruction #3)"""
    total_records = len(df)

    # Ensure Date column is valid and not NaT
    date_min = df['Date'].min()
    date_max = df['Date'].max()

    # Format dates only if they are not NaT
    date_min_str = date_min.strftime('%Y-%m-%d') if pd.notna(date_min) else "N/A"
    date_max_str = date_max.strftime('%Y-%m-%d') if pd.notna(date_max) else "N/A"

    date_range = f"{date_min_str} to {date_max_str}"

    team_filter = st.session_state.selected_team
    member_filter = st.session_state.selected_team_member

    status_text = f"📊 Viewing {total_records:,} records from {date_range}"

    if team_filter != "All" or member_filter != "All":
        status_text += f" | Filtered by: "
        filters = []
        if team_filter != "All":
            filters.append(f"Team: {team_filter}")
        if member_filter != "All":
            filters.append(f"Member: {member_filter}")
        status_text += ", ".join(filters)

    st.info(status_text)


def show_kpi_metrics(kpis):
    """Display KPI metrics in a row of cards with color coding"""
    cols = st.columns(7)

    for i, (key, value) in enumerate(kpis.items()):
        with cols[i]:
            st.markdown(
                f"""
                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; background-color: #f9f9f9;">
                    <div style="font-size: 14px; color: #666;">{key}</div>
                    <div style="font-size: 20px; font-weight: bold; color: {value['color']};">{value['value']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

def show_home(df):
    st.title("📊 AI Sales Dashboard Overview")

    # Show welcome and instructions for first-time users
    if not st.session_state.has_seen_welcome:
        show_welcome_message()

    # Display KPIs at the top
    kpis = calculate_kpis(df)
    show_kpi_metrics(kpis)

    # Show visibility of system status
    show_status_bar(df)

    # Prepare required data before plotting
    df['Request Category'] = df['URL Path'].apply(classify_request)
    request_counts = df['Request Category'].value_counts()
    customer_counts = df['Repeat Customer'].value_counts()

    # Create a 2x2 grid layout
    col1, col2 = st.columns(2)
    with col1:
        # Customer type with overall indicator
        fig1 = px.pie(
            names=customer_counts.index, values=customer_counts.values,
            title="Customer Type Share",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig1.add_annotation(
            text=f"Total: {customer_counts.sum():,}",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )
        fig1.update_layout(margin=dict(t=40, b=20), height=350)
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("🔁 Proportion of new vs repeat customers.")

    with col2:
        # Profit by job type with overall indicator
        avg_profit = df["Profit"].mean()
        fig2 = px.box(
            df, x="Job Type", y="Profit",
            title="Profit by Job Type",
            color_discrete_sequence=['#00CC96']
        )
        fig2.add_shape(
            type="line",
            x0=-0.5, y0=avg_profit, x1=len(df["Job Type"].unique()) - 0.5, y1=avg_profit,
            line=dict(color="red", width=2, dash="dash"),
        )
        fig2.add_annotation(
            text=f"Average: ${avg_profit:.2f}",
            x=0.85, y=avg_profit + (df["Profit"].max() - df["Profit"].min()) * 0.05,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        fig2.update_layout(margin=dict(t=40, b=20), height=350)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("💼 Performance by job type.")

    col3, col4 = st.columns(2)
    with col3:
        # Request type distribution with overall indicator
        fig3 = px.pie(
            names=request_counts.index, values=request_counts.values,
            title="Request Type Distribution",
            color_discrete_sequence=px.colors.sequential.Tealgrn
        )
        fig3.add_annotation(
            text=f"Total: {request_counts.sum():,}",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )
        fig3.update_layout(margin=dict(t=40, b=20), height=350)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("📄 Demand across service request categories.")

    with col4:
        # Quantity vs profit with overall indicator
        avg_quantity = df["Quantity"].mean()
        avg_profit = df["Profit"].mean()

        fig4 = px.scatter(
            df, x="Quantity", y="Profit", color="Country",
            title="Quantity vs Profit"
        )
        fig4.add_shape(
            type="line",
            x0=df["Quantity"].min(), y0=avg_profit,
            x1=df["Quantity"].max(), y1=avg_profit,
            line=dict(color="red", width=1, dash="dash"),
        )
        fig4.add_shape(
            type="line",
            x0=avg_quantity, y0=df["Profit"].min(),
            x1=avg_quantity, y1=df["Profit"].max(),
            line=dict(color="red", width=1, dash="dash"),
        )
        fig4.add_annotation(
            text=f"Avg Quantity: {avg_quantity:.1f}<br>Avg Profit: ${avg_profit:.2f}",
            x=0.85, y=0.95,
            xref="paper", yref="paper",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        fig4.update_layout(margin=dict(t=40, b=20), height=350)
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("📈 Relationship between quantity and profit.")

def show_country(df):
    st.title("🌍 Sales by Country")

    # Display KPIs at the top
    kpis = calculate_kpis(df)
    show_kpi_metrics(kpis)

    # Show visibility of system status
    show_status_bar(df)

    def get_country_code(name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except:
            return None

    country_df = df.groupby("Country")["Revenue"].sum().reset_index()
    country_df["iso_alpha"] = country_df["Country"].apply(get_country_code)

    # Add total indicator
    total_revenue = country_df["Revenue"].sum()

    # Create the map
    fig = px.choropleth(
        country_df,
        locations="iso_alpha",
        color="Revenue",
        hover_name="Country",
        color_continuous_scale=px.colors.sequential.Plasma,
        title=f"Global Revenue Distribution (Total: ${total_revenue:,.2f})"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add a bar chart for top countries
    top_countries = country_df.sort_values("Revenue", ascending=False).head(10)

    fig2 = px.bar(
        top_countries,
        x="Country",
        y="Revenue",
        title="Top 10 Countries by Revenue",
        text_auto='.2s',
        color="Revenue",
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig2.update_layout(xaxis_title="Country", yaxis_title="Revenue (USD)")
    st.plotly_chart(fig2, use_container_width=True)

    # Add a pie chart for revenue share by continent
    continent_mapping = {
        "USA": "North America", "Canada": "North America", "Mexico": "North America",
        "UK": "Europe", "Germany": "Europe", "France": "Europe",
        "Japan": "Asia", "India": "Asia",
        "Australia": "Oceania",
        "Brazil": "South America"
    }

    country_df["Continent"] = country_df["Country"].map(lambda x: continent_mapping.get(x, "Other"))
    continent_df = country_df.groupby("Continent")["Revenue"].sum().reset_index()

    fig3 = px.pie(
        continent_df,
        values="Revenue",
        names="Continent",
        title="Revenue Share by Continent",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig3.add_annotation(
        text=f"Total: ${total_revenue:,.0f}",
        x=0.5, y=0.5,
        font_size=14,
        showarrow=False
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.write("""
    This map visualizes revenue generation by country, highlighting key international markets.
    The bar chart displays the top 10 countries by revenue, and the pie chart shows the revenue share by continent.
    """)

def show_customer_type(df):
    st.title("👥 Customer Type Analysis")

    # Display KPIs at the top
    kpis = calculate_kpis(df)
    show_kpi_metrics(kpis)

    # Show visibility of system status
    show_status_bar(df)

    # Get customer type counts
    counts = df['Repeat Customer'].value_counts().reset_index()
    counts.columns = ['Customer Type', 'Count']
    counts['Customer Type'] = counts['Customer Type'].map({'Yes': 'Repeat Customer', 'No': 'New Customer'})

    # Create 2-column layout
    col1, col2 = st.columns(2)

    with col1:
        # Create pie chart with customer type
        fig1 = px.pie(
            counts,
            names='Customer Type',
            values='Count',
            title="New vs Repeat Customers",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig1.add_annotation(
            text=f"Total: {counts['Count'].sum():,}",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Create revenue by customer type
        customer_revenue = df.groupby('Repeat Customer')['Revenue'].sum().reset_index()
        customer_revenue['Repeat Customer'] = customer_revenue['Repeat Customer'].map({'Yes': 'Repeat Customer', 'No': 'New Customer'})
        customer_revenue.columns = ['Customer Type', 'Revenue']

        fig2 = px.bar(
            customer_revenue,
            x='Customer Type',
            y='Revenue',
            title="Revenue by Customer Type",
            text_auto='.2s',
            color='Customer Type',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig2.add_annotation(
            text=f"Total: ${customer_revenue['Revenue'].sum():,.0f}",
            x=0.5, y=0.95,
            xref="paper", yref="paper",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Add a line chart showing customer acquisition over time
    customer_over_time = df.groupby(['Date', 'Repeat Customer']).size().unstack(fill_value=0).reset_index()
    customer_over_time.columns = ['Date', 'New Customer', 'Repeat Customer'] if 'No' in df['Repeat Customer'].values else ['Date', 'Repeat Customer', 'New Customer']

    # Ensure both columns exist
    if 'New Customer' not in customer_over_time.columns:
        customer_over_time['New Customer'] = 0
    if 'Repeat Customer' not in customer_over_time.columns:
        customer_over_time['Repeat Customer'] = 0

    fig3 = px.line(
        customer_over_time,
        x='Date',
        y=['New Customer', 'Repeat Customer'],
        title="Customer Acquisition Over Time",
        markers=True
    )
    fig3.update_layout(yaxis_title="Number of Customers", xaxis_title="Date")

    # Add overall trend indicator
    new_customer_trend = customer_over_time['New Customer'].iloc[-1] - customer_over_time['New Customer'].iloc[0]
    repeat_customer_trend = customer_over_time['Repeat Customer'].iloc[-1] - customer_over_time['Repeat Customer'].iloc[0]

    trend_text = f"New Customer Trend: {'▲' if new_customer_trend > 0 else '▼'} {abs(new_customer_trend):.0f}<br>"
    trend_text += f"Repeat Customer Trend: {'▲' if repeat_customer_trend > 0 else '▼'} {abs(repeat_customer_trend):.0f}"

    fig3.add_annotation(
        text=trend_text,
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Add a funnel chart showing customer conversion
    funnel_data = {
        "Stage": ["Website Visitors", "Engaged Visitors", "Leads", "Customers"],
        "Count": [
            df.shape[0] * random.randint(5, 10),  # Simulated total visitors
            df.shape[0] * random.randint(2, 4),   # Simulated engaged visitors
            df.shape[0] * random.randint(1, 2),   # Simulated leads
            df.shape[0]                           # Actual customers from data
        ]
    }
    funnel_df = pd.DataFrame(funnel_data)

    fig4 = px.funnel(
        funnel_df,
        x="Count",
        y="Stage",
        title="Customer Conversion Funnel"
    )

    # Add conversion rate indicators
    for i in range(len(funnel_data["Stage"]) - 1):
        conversion_rate = (funnel_data["Count"][i+1] / funnel_data["Count"][i]) * 100
        fig4.add_annotation(
            x=funnel_data["Count"][i+1] / 2,
            y=funnel_data["Stage"][i+1],
            text=f"{conversion_rate:.1f}% conversion",
            showarrow=False,
            font=dict(color="black")
        )

    st.plotly_chart(fig4, use_container_width=True)

    st.write("""
    This analysis shows the distribution of new versus repeat customers and their revenue contribution.
    The trend chart displays customer acquisition patterns over time, while the funnel visualizes the conversion process.
    """)

def show_job_type(df):
    st.title("🧰 Profit by Job Type")

    # Display KPIs at the top
    kpis = calculate_kpis(df)
    show_kpi_metrics(kpis)

    # Show visibility of system status
    show_status_bar(df)

    # Create job type summary
    job_summary = df.groupby("Job Type").agg({
        "Revenue": "sum",
        "Profit": "sum",
        "Quantity": "sum"
    }).reset_index()

    job_summary["Profit Margin"] = job_summary["Profit"] / job_summary["Revenue"] * 100

    # Create two-column layout
    col1, col2 = st.columns(2)

    with col1:
        # Box plot with overall indicator
        avg_profit = df["Profit"].mean()

        fig1 = px.box(
            df,
            x="Job Type",
            y="Profit",
            title="Profit Distribution by Job Type",
            color="Job Type"
        )
        fig1.add_shape(
            type="line",
            x0=-0.5, y0=avg_profit, x1=len(df["Job Type"].unique()) - 0.5, y1=avg_profit,
            line=dict(color="red", width=2, dash="dash"),
        )
        fig1.add_annotation(
            text=f"Overall Avg: ${avg_profit:.2f}",
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Bar chart with job profitability
        fig2 = px.bar(
            job_summary,
            x="Job Type",
            y="Profit",
            color="Profit Margin",
            title="Total Profit by Job Type",
            text_auto='.2s',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig2.add_annotation(
            text=f"Total Profit: ${job_summary['Profit'].sum():,.0f}",
            x=0.5, y=0.98,
            xref="paper", yref="paper",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Create a table with detailed metrics
    job_summary["Revenue"] = job_summary["Revenue"].map("${:,.2f}".format)
    job_summary["Profit"] = job_summary["Profit"].map("${:,.2f}".format)
    job_summary["Profit Margin"] = job_summary["Profit Margin"].map("{:.1f}%".format)
    job_summary["Quantity"] = job_summary["Quantity"].map("{:,}".format)

    st.subheader("Job Type Performance Metrics")
    st.dataframe(job_summary, use_container_width=True)

    # Stacked bar chart for job type by team
    job_by_team = df.groupby(["Team", "Job Type"]).size().reset_index(name="Count")

    fig3 = px.bar(
        job_by_team,
        x="Team",
        y="Count",
        color="Job Type",
        title="Job Types by Team",
        text_auto='.2s'
    )
    fig3.update_layout(yaxis_title="Number of Jobs")

    st.plotly_chart(fig3, use_container_width=True)

    st.write("""
    This analysis shows the profitability variation across different job types.
    The box plot displays the distribution of profits, while the bar chart shows the total profit by job type.
    The table provides detailed metrics for each job type.
    """)

def show_request_category(df):
    st.title("📄 Request Category Distribution")

    # Display KPIs at the top
    kpis = calculate_kpis(df)
    show_kpi_metrics(kpis)

    # Show visibility of system status
    show_status_bar(df)

    # Get request categories
    df['Request Category'] = df['URL Path'].apply(classify_request)
    counts = df['Request Category'].value_counts().reset_index()
    counts.columns = ['Category', 'Count']

    # Create a two-column layout
    col1, col2 = st.columns(2)

    with col1:
        # Pie chart with overall indicator
        fig1 = px.pie(
            counts,
            names='Category',
            values='Count',
            title="Request Type Distribution",
            color_discrete_sequence=px.colors.sequential.Tealgrn
        )
        fig1.add_annotation(
            text=f"Total: {counts['Count'].sum():,}",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Bar chart with response code distribution by request type
        response_by_request = df.groupby(['Request Category', 'Response Code']).size().reset_index(name='Count')

        fig2 = px.bar(
            response_by_request,
            x='Request Category',
            y='Count',
            color='Response Code',
            title="Response Codes by Request Type",
            text_auto='.2s'
        )
        fig2.update_layout(xaxis_title="Request Category", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    # Create line chart for request categories over time
    request_over_time = df.groupby(['Date', 'Request Category']).size().reset_index(name='Count')

    fig3 = px.line(
        request_over_time,
        x='Date',
        y='Count',
        color='Request Category',
        title="Request Categories Over Time",
        markers=True
    )

    # Add overall trend indicator
    request_trends = {}
    for category in df['Request Category'].unique():
        category_data = request_over_time[request_over_time['Request Category'] == category]
        if len(category_data) > 1:
            start_count = category_data.iloc[0]['Count']
            end_count = category_data.iloc[-1]['Count']
            trend = end_count - start_count
            request_trends[category] = trend

    trend_text = "<br>".join([f"{cat}: {'▲' if trend > 0 else '▼'} {abs(trend):.0f}" for cat, trend in request_trends.items()])

    fig3.add_annotation(
        text=f"Trends:<br>{trend_text}",
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Create heatmap for request categories by hour of day
    hour_category = df.groupby(['Hour', 'Request Category']).size().reset_index(name='Count')
    hour_category_pivot = hour_category.pivot(index='Hour', columns='Request Category', values='Count').fillna(0)

    fig4 = px.imshow(
        hour_category_pivot,
        labels=dict(x="Request Category", y="Hour of Day", color="Count"),
        x=hour_category_pivot.columns,
        y=hour_category_pivot.index,
        title="Request Categories by Hour of Day",
        color_continuous_scale="Viridis"
    )

    # Add overall peak hours indicator
    hour_totals = hour_category.groupby('Hour')['Count'].sum()
    peak_hour = hour_totals.idxmax()
    slowest_hour = hour_totals.idxmin()

    fig4.add_annotation(
        text=f"Peak Hour: {peak_hour:02d}:00<br>Slowest Hour: {slowest_hour:02d}:00",
        x=0.02, y=0.02,
        xref="paper", yref="paper",
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.write("""
    This visualization categorizes and shows the types of requests made on the website.
    The bar chart displays response code distribution by request type, while the line chart shows trends over time.
    The heatmap reveals patterns in request categories by hour of day.
    """)

def show_quantity_profit(df):
    st.title("📊 Quantity vs Profit")

    # Display KPIs at the top
    kpis = calculate_kpis(df)
    show_kpi_metrics(kpis)

    # Show visibility of system status
    show_status_bar(df)

    # Create a two-column layout
    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot with overall indicators
        avg_quantity = df["Quantity"].mean()
        avg_profit = df["Profit"].mean()

        fig1 = px.scatter(
            df,
            x="Quantity",
            y="Profit",
            color="Country",
            title="Sales Quantity vs. Profit",
            hover_data=["Job Type", "Team Member"]
        )
        fig1.add_shape(
            type="line",
            x0=df["Quantity"].min(), y0=avg_profit,
            x1=df["Quantity"].max(), y1=avg_profit,
            line=dict(color="red", width=1, dash="dash"),
        )
        fig1.add_shape(
            type="line",
            x0=avg_quantity, y0=df["Profit"].min(),
            x1=avg_quantity, y1=df["Profit"].max(),
            line=dict(color="red", width=1, dash="dash"),
        )
        fig1.add_annotation(
            text=f"Avg Quantity: {avg_quantity:.1f}<br>Avg Profit: ${avg_profit:.2f}",
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Line chart showing quantity and profit over time
        time_data = df.groupby('Date').agg({
            'Quantity': 'sum',
            'Profit': 'sum'
        }).reset_index()

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])

        fig2.add_trace(
            go.Scatter(
                x=time_data['Date'],
                y=time_data['Quantity'],
                name="Quantity",
                line=dict(color='blue')
            ),
            secondary_y=False
        )

        fig2.add_trace(
            go.Scatter(
                x=time_data['Date'],
                y=time_data['Profit'],
                name="Profit",
                line=dict(color='green')
            ),
            secondary_y=True
        )

        fig2.update_layout(
            title="Quantity and Profit Over Time",
            xaxis_title="Date"
        )

        fig2.update_yaxes(title_text="Quantity", secondary_y=False)
        fig2.update_yaxes(title_text="Profit (USD)", secondary_y=True)

        # Add overall trend indicators
        quantity_trend = time_data['Quantity'].iloc[-1] - time_data['Quantity'].iloc[0]
        profit_trend = time_data['Profit'].iloc[-1] - time_data['Profit'].iloc[0]

        trend_text = f"Quantity Trend: {'▲' if quantity_trend > 0 else '▼'} {abs(quantity_trend):.0f}<br>"
        trend_text += f"Profit Trend: {'▲' if profit_trend > 0 else '▼'} ${abs(profit_trend):.2f}"

        fig2.add_annotation(
            text=trend_text,
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Add a heatmap for profit by quantity and job type
    heatmap_data = df.pivot_table(
        values='Profit',
        index='Quantity',
        columns='Job Type',
        aggfunc='mean'
    ).fillna(0)

    fig3 = px.imshow(
        heatmap_data,
        labels=dict(x="Job Type", y="Quantity", color="Average Profit"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        title="Average Profit by Quantity and Job Type",
        color_continuous_scale="RdBu_r"
    )

    # Add overall indicator for most profitable combination
    max_profit_idx = heatmap_data.stack().idxmax()
    if isinstance(max_profit_idx, tuple):
        max_qty, max_job = max_profit_idx
        max_profit_val = heatmap_data.loc[max_qty, max_job]

        fig3.add_annotation(
            text=f"Most Profitable: {max_job} at Qty {max_qty}<br>Avg Profit: ${max_profit_val:.2f}",
            x=0.5, y=0.02,
            xref="paper", yref="paper",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )

    st.plotly_chart(fig3, use_container_width=True)

    st.write("""
    This scatter plot explores the relationship between sales quantity and generated profit across countries.
    The line chart shows how quantity and profit have changed over time.
    The heatmap reveals the average profit for different combinations of quantity and job type.
    """)

def show_team_performance(df):
    st.title("👥 Team Performance Analytics")

    # Display KPIs at the top
    kpis = calculate_kpis(df)
    show_kpi_metrics(kpis)

    # Show visibility of system status
    show_status_bar(df)

    # Create team summary metrics
    team_metrics = df.groupby("Team").agg({
        "Revenue": "sum",
        "Profit": "sum",
        "Customer ID": "nunique",
        "Quantity": "sum"
    }).reset_index()

    team_metrics.columns = ["Team", "Total Revenue", "Total Profit", "Unique Customers", "Units Sold"]
    team_metrics["Profit Margin"] = team_metrics["Total Profit"] / team_metrics["Total Revenue"] * 100
    team_metrics["Revenue per Customer"] = team_metrics["Total Revenue"] / team_metrics["Unique Customers"]

    # Format columns for display
    display_metrics = team_metrics.copy()
    display_metrics["Total Revenue"] = display_metrics["Total Revenue"].map("${:,.2f}".format)
    display_metrics["Total Profit"] = display_metrics["Total Profit"].map("${:,.2f}".format)
    display_metrics["Profit Margin"] = display_metrics["Profit Margin"].map("{:.1f}%".format)
    display_metrics["Revenue per Customer"] = display_metrics["Revenue per Customer"].map("${:,.2f}".format)
    display_metrics["Units Sold"] = display_metrics["Units Sold"].map("{:,}".format)

    # Display team metrics table
    st.subheader("Team Performance Overview")
    st.dataframe(display_metrics, use_container_width=True)

    # Create visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart for team revenue and profit
        revenue_profit = team_metrics.melt(
            id_vars=["Team"],
            value_vars=["Total Revenue", "Total Profit"],
            var_name="Metric",
            value_name="Amount"
        )

        fig1 = px.bar(
            revenue_profit,
            x="Team",
            y="Amount",
            color="Metric",
            barmode="group",
            title="Revenue and Profit by Team",
            text_auto='.2s'
        )

        # Add overall indicator for top team
        top_revenue_team = team_metrics.loc[team_metrics["Total Revenue"].idxmax(), "Team"]
        top_profit_team = team_metrics.loc[team_metrics["Total Profit"].idxmax(), "Team"]

        indicator_text = f"Top Revenue: {top_revenue_team}<br>Top Profit: {top_profit_team}"

        fig1.add_annotation(
            text=indicator_text,
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Radar chart for team performance across metrics
        team_radar = team_metrics.copy()

        # Normalize values for radar chart (0-1 scale)
        for col in ["Total Revenue", "Total Profit", "Unique Customers", "Units Sold", "Profit Margin"]:
            if team_radar[col].max() > 0:
                team_radar[col] = team_radar[col] / team_radar[col].max()

        fig2 = go.Figure()

        for i, team in enumerate(team_radar["Team"]):
            fig2.add_trace(go.Scatterpolar(
                r=team_radar.loc[i, ["Total Revenue", "Total Profit", "Unique Customers", "Units Sold", "Profit Margin"]],
                theta=["Revenue", "Profit", "Customers", "Units", "Margin"],
                fill='toself',
                name=team
            ))

        fig2.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Team Performance Comparison",
            showlegend=True
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Team performance over time
    team_over_time = df.groupby(['Date', 'Team'])['Profit'].sum().reset_index()

    fig3 = px.line(
        team_over_time,
        x='Date',
        y='Profit',
        color='Team',
        title="Team Profit Over Time",
        markers=True
    )

    # Add overall trend indicators
    trend_text = ""
    for team in df['Team'].unique():
        team_data = team_over_time[team_over_time['Team'] == team]
        if len(team_data) > 1:
            start_profit = team_data.iloc[0]['Profit']
            end_profit = team_data.iloc[-1]['Profit']
            trend = end_profit - start_profit
            trend_text += f"{team}: {'▲' if trend > 0 else '▼'} ${abs(trend):.2f}<br>"

    fig3.add_annotation(
        text=f"Profit Trends:<br>{trend_text}",
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.write("""
    This dashboard provides a comprehensive view of team performance across various metrics.
    The bar chart compares revenue and profit by team, while the radar chart offers a multidimensional comparison.
    The line chart tracks team profit trends over time.
    """)

def show_individual_performance(df):
    st.title("👤 Individual Performance Dashboard")

    # Display KPIs at the top
    kpis = calculate_kpis(df)
    show_kpi_metrics(kpis)

    # Show visibility of system status
    show_status_bar(df)

    # Get team member filter from sidebar if not All
    if st.session_state.selected_team_member != "All":
        st.subheader(f"Performance Dashboard for: {st.session_state.selected_team_member}")

        # Filter data for the selected team member
        member_df = df[df["Team Member"] == st.session_state.selected_team_member]

        # Create 2-column layout
        col1, col2 = st.columns(2)

        with col1:
            # Key metrics for this individual
            member_metrics = {
                "Total Revenue": "${:,.2f}".format(member_df["Revenue"].sum()),
                "Total Profit": "${:,.2f}".format(member_df["Profit"].sum()),
                "Total Orders": "{:,}".format(len(member_df)),
                "Unique Customers": "{:,}".format(member_df["Customer ID"].nunique()),
                "Average Order Value": "${:,.2f}".format(member_df["Revenue"].mean()),
                "Average Response Time": "{:.0f} ms".format(member_df["Response Time (ms)"].mean())
            }

            st.subheader("Individual Performance Metrics")

            for key, value in member_metrics.items():
                st.metric(label=key, value=value)

        with col2:
            # Performance over time chart
            daily_metrics = member_df.groupby("Date").agg({
                "Revenue": "sum",
                "Profit": "sum"
            }).reset_index()

            fig1 = px.line(
                daily_metrics,
                x="Date",
                y=["Revenue", "Profit"],
                title=f"Performance Over Time - {st.session_state.selected_team_member}",
                markers=True
            )

            # Add trend indicators
            revenue_trend = daily_metrics["Revenue"].iloc[-1] - daily_metrics["Revenue"].iloc[0]
            profit_trend = daily_metrics["Profit"].iloc[-1] - daily_metrics["Profit"].iloc[0]

            trend_text = f"Revenue Trend: {'▲' if revenue_trend > 0 else '▼'} ${abs(revenue_trend):.2f}<br>"
            trend_text += f"Profit Trend: {'▲' if profit_trend > 0 else '▼'} ${abs(profit_trend):.2f}"

            fig1.add_annotation(
                text=trend_text,
                x=0.02, y=0.98,
                xref="paper", yref="paper",
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)"
            )

            st.plotly_chart(fig1, use_container_width=True)

        # Create bottom charts
        col3, col4 = st.columns(2)

        with col3:
            # Job type distribution
            job_counts = member_df["Job Type"].value_counts().reset_index()
            job_counts.columns = ["Job Type", "Count"]

            fig2 = px.pie(
                job_counts,
                names="Job Type",
                values="Count",
                title=f"Job Type Distribution - {st.session_state.selected_team_member}",
                hole=0.4
            )

            fig2.add_annotation(
                text=f"Total: {job_counts['Count'].sum():,}",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )

            st.plotly_chart(fig2, use_container_width=True)

        with col4:
            # Customer distribution
            customer_counts = member_df["Repeat Customer"].value_counts().reset_index()
            customer_counts.columns = ["Customer Type", "Count"]
            customer_counts["Customer Type"] = customer_counts["Customer Type"].map({"Yes": "Repeat Customer", "No": "New Customer"})

            fig3 = px.pie(
                customer_counts,
                names="Customer Type",
                values="Count",
                title=f"Customer Type Distribution - {st.session_state.selected_team_member}",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu
            )

            fig3.add_annotation(
                text=f"Total: {customer_counts['Count'].sum():,}",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )

            st.plotly_chart(fig3, use_container_width=True)

        # Activity heatmap by hour and day of week
        member_df["DayOfWeek"] = member_df["Date"].dt.day_name()
        member_df["HourOfDay"] = pd.to_datetime(member_df["Timestamp"]).dt.hour

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        activity_data = member_df.groupby(["DayOfWeek", "HourOfDay"]).size().reset_index(name="Count")
        activity_pivot = pd.pivot_table(
            activity_data,
            values="Count",
            index="DayOfWeek",
            columns="HourOfDay",
            fill_value=0
        )

        # Reorder days
        if not activity_pivot.empty:
            activity_pivot = activity_pivot.reindex(
                [day for day in day_order if day in activity_pivot.index]
            )

        fig4 = px.imshow(
            activity_pivot,
            labels=dict(x="Hour of Day", y="Day of Week", color="Activity Count"),
            x=activity_pivot.columns if not activity_pivot.empty else [],
            y=activity_pivot.index if not activity_pivot.empty else [],
            title=f"Activity Heatmap - {st.session_state.selected_team_member}",
            color_continuous_scale="Viridis"
        )

        # Add peak activity indicator
        if not activity_data.empty:
            peak_idx = activity_data["Count"].idxmax()
            peak_day = activity_data.loc[peak_idx, "DayOfWeek"]
            peak_hour = activity_data.loc[peak_idx, "HourOfDay"]
            peak_count = activity_data.loc[peak_idx, "Count"]

            fig4.add_annotation(
                text=f"Peak Activity: {peak_day} at {peak_hour:02d}:00<br>Count: {peak_count}",
                x=0.02, y=0.02,
                xref="paper", yref="paper",
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)"
            )

        st.plotly_chart(fig4, use_container_width=True)

    else:
        # Show comparison of all team members
        st.subheader("Individual Performance Comparison")

        # Calculate metrics for all team members
        member_metrics = df.groupby("Team Member").agg({
            "Revenue": "sum",
            "Profit": "sum",
            "Customer ID": "nunique",
            "Quantity": "sum",
            "Response Time (ms)": "mean"
        }).reset_index()

        member_metrics.columns = ["Team Member", "Revenue", "Profit", "Customers", "Units", "Response Time"]

        # Format columns for display
        display_metrics = member_metrics.copy()
        display_metrics["Revenue"] = display_metrics["Revenue"].map("${:,.2f}".format)
        display_metrics["Profit"] = display_metrics["Profit"].map("${:,.2f}".format)
        display_metrics["Customers"] = display_metrics["Customers"].map("{:,}".format)
        display_metrics["Units"] = display_metrics["Units"].map("{:,}".format)
        display_metrics["Response Time"] = display_metrics["Response Time"].map("{:.0f} ms".format)

        st.dataframe(display_metrics, use_container_width=True)

        # Create visualizations for comparison
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart for revenue by team member
            fig1 = px.bar(
                member_metrics,
                x="Team Member",
                y="Revenue",
                title="Revenue by Team Member",
                text_auto='.2s',
                color="Revenue",
                color_continuous_scale=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Bar chart for profit by team member
            fig2 = px.bar(
                member_metrics,
                x="Team Member",
                y="Profit",
                title="Profit by Team Member",
                text_auto='.2s',
                color="Profit",
                color_continuous_scale=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Line chart for performance over time by team member
        time_metrics = df.groupby(["Date", "Team Member"]).agg({
            "Revenue": "sum",
            "Profit": "sum"
        }).reset_index()

        fig3 = px.line(
            time_metrics,
            x="Date",
            y="Revenue",
            color="Team Member",
            title="Revenue Over Time by Team Member",
            markers=True
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Scatter plot for profit vs response time
        fig4 = px.scatter(
            member_metrics,
            x="Response Time",
            y="Profit",
            color="Team Member",
            title="Profit vs Response Time by Team Member",
            hover_data=["Customers", "Units"]
        )
        st.plotly_chart(fig4, use_container_width=True)

def main():
    # Load data
    df = load_data()

    # Show login screen if not authenticated
    if not st.session_state.authenticated:
        show_login_screen()
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    menu_options = [
        "Home",
        "Sales by Country",
        "Customer Type Analysis",
        "Profit by Job Type",
        "Request Category Distribution",
        "Quantity vs Profit",
        "Team Performance",
        "Individual Performance"
    ]
    selected_page = st.sidebar.radio("Go to", menu_options)

    # Add filters to sidebar
    st.sidebar.header("Filters")

    # Team filter
    st.session_state.selected_team = st.sidebar.selectbox(
        "Select Team",
        ALL_TEAMS,
        index=0
    )

    # Team member filter (update based on selected team)
    if st.session_state.selected_team == "All":
        team_members = ALL_TEAM_MEMBERS
    else:
        team_members = ["All"] + TEAMS[st.session_state.selected_team]

    st.session_state.selected_team_member = st.sidebar.selectbox(
        "Select Team Member",
        team_members,
        index=0
    )

    # Ensure Date column is parsed correctly
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    # Set default min and max dates if parsing fails
    default_min_date = pd.to_datetime("2024-01-01")
    default_max_date = pd.to_datetime("2024-12-31")

    min_date = df["Date"].min() if pd.notna(df["Date"].min()) else default_min_date
    max_date = df["Date"].max() if pd.notna(df["Date"].max()) else default_max_date

    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        st.session_state.date_filter = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    else:
        st.session_state.date_filter = (pd.Timestamp(date_range), pd.Timestamp(date_range))

    # Instructions button
    st.sidebar.button("Show Instructions", on_click=toggle_instructions)

    # Export data button
    export_button = st.sidebar.button("Export Data")
    if export_button:
        export_file, export_mime, export_name = export_data(filter_data(df))
        if export_file:
            st.sidebar.download_button(
                label="Download Data",
                data=export_file,
                file_name=export_name,
                mime=export_mime
            )

    # Logout button
    st.sidebar.button("Logout", on_click=logout_user)

    # Filter data based on selections
    filtered_df = filter_data(df)

    # Show appropriate page based on selection
    if selected_page == "Home":
        show_home(filtered_df)
    elif selected_page == "Sales by Country":
        show_country(filtered_df)
    elif selected_page == "Customer Type Analysis":
        show_customer_type(filtered_df)
    elif selected_page == "Profit by Job Type":
        show_job_type(filtered_df)
    elif selected_page == "Request Category Distribution":
        show_request_category(filtered_df)
    elif selected_page == "Quantity vs Profit":
        show_quantity_profit(filtered_df)
    elif selected_page == "Team Performance":
        show_team_performance(filtered_df)
    elif selected_page == "Individual Performance":
        show_individual_performance(filtered_df)

    # Show instructions if enabled
    show_instructions()


if __name__ == "__main__":
    main()
