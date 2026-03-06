from flask import Flask, request, render_template_string
import numpy as np

app = Flask(__name__)

# ========== MODEL COEFFICIENTS FROM YOUR TRAINED MODEL ==========
MODEL = {
    'intercept': -15095881.344720656,
    'coefficients': [
        7481.167570659545,    # work_year
        12531.902273660227,   # experience_level
        -14824.301873756254,  # employment_type
        1252.228415892946,    # job_title
        489.70610369674796,   # company_location
        1148.627350683757,    # employee_residence
        108.99533311993673,   # remote_ratio
        -11628.470924001436   # company_size
    ]
}

# ========== COMPLETE HTML TEMPLATE ==========
HTML_TEMPLATE = '''

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SalaryPredict • AI-Powered Salary Predictor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0c10;
            color: #ffffff;
            line-height: 1.6;
            overflow-x: hidden;
            position: relative;
        }

        /* ========== ANIMATED BACKGROUND CANVAS ========== */
        #canvas-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
        }

        /* ========== GLASS EFFECT OVERLAY ========== */
        .glass-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 50% 50%, rgba(10, 12, 16, 0.3) 0%, rgba(10, 12, 16, 0.8) 100%);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            z-index: 1;
            pointer-events: none;
        }

        /* ========== MAIN CONTENT ========== */
        .content-wrapper {
            position: relative;
            z-index: 2;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Navigation */
        .navbar {
            display: flex;
            align-items: center;
            padding: 1.5rem 4rem;
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-right: 3rem;
            letter-spacing: -0.5px;
        }

        .nav-links {
            display: flex;
            gap: 2.5rem;
        }

        .nav-links a {
            text-decoration: none;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 500;
            font-size: 0.95rem;
            transition: color 0.3s;
            cursor: pointer;
        }

        .nav-links a:hover,
        .nav-links a.active {
            color: #60a5fa;
        }

        .powered-by {
            margin-left: auto;
            font-size: 0.85rem;
            color: rgba(255, 255, 255, 0.4);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .powered-by span {
            color: rgba(255, 255, 255, 0.9);
            font-weight: 500;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            padding: 0.2rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
        }

        /* Main Container */
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem 4rem;
            position: relative;
            z-index: 10;
            flex: 1;
        }

        /* Hero Section */
        .hero {
            margin: 3rem 0 4rem 0;
            text-align: center;
        }

        .hero h1 {
            font-size: 4rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }

        .hero h1 span {
            background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            display: inline-block;
        }

        .hero p {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.6);
            max-width: 700px;
            margin: 0 auto 2.5rem;
        }

        .hero-buttons {
            display: flex;
            gap: 1rem;
            justify-content: center;
        }

        .primary-btn {
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            color: white;
            border: none;
            padding: 1rem 2.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
            box-shadow: 0 10px 30px -10px rgba(96, 165, 250, 0.3);
        }

        .primary-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 40px -10px rgba(96, 165, 250, 0.5);
        }

        .secondary-btn {
            background: rgba(255, 255, 255, 0.05);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1rem 2.5rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .secondary-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }

        /* Stats Cards */
        .stats-container {
            display: flex;
            gap: 2rem;
            margin: 4rem 0;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 24px;
            padding: 2rem;
            flex: 1;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: transform 0.3s;
            text-align: center;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(96, 165, 250, 0.2);
        }

        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }

        .stat-label {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Section Styles */
        .section {
            margin: 6rem 0;
            scroll-margin-top: 100px;
        }

        .section-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-align: center;
            background: linear-gradient(135deg, #ffffff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .section-subtitle {
            text-align: center;
            color: rgba(255, 255, 255, 0.5);
            margin-bottom: 3rem;
            font-size: 1.1rem;
        }

        /* How It Works Section */
        .steps-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
        }

        .step-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 24px;
            padding: 2.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }

        .step-card:hover {
            transform: translateY(-5px);
            border-color: rgba(96, 165, 250, 0.3);
            background: rgba(255, 255, 255, 0.03);
        }

        .step-number {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, rgba(96, 165, 250, 0.2), rgba(167, 139, 250, 0.2));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
            line-height: 1;
        }

        .step-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: white;
        }

        .step-description {
            color: rgba(255, 255, 255, 0.5);
            line-height: 1.6;
        }

        /* About Section */
        .about-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4rem;
            align-items: center;
        }

        .about-text {
            color: rgba(255, 255, 255, 0.7);
        }

        .about-text p {
            margin-bottom: 1.5rem;
            font-size: 1.1rem;
        }

        .feature-list {
            list-style: none;
            margin-top: 2rem;
        }

        .feature-list li {
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .feature-icon {
            width: 24px;
            height: 24px;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 14px;
        }

        .about-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }

        .about-stat-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .about-stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: #60a5fa;
            margin-bottom: 0.5rem;
        }

        .about-stat-label {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.9rem;
        }

        /* Prediction Section */
        .prediction-section {
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 32px;
            padding: 3rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin: 3rem 0;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }

        .form-group {
            margin-bottom: 1rem;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            display: block;
            margin-bottom: 0.5rem;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 500;
            font-size: 0.9rem;
            letter-spacing: 0.3px;
        }

        select {
            width: 100%;
            padding: 1rem 1.2rem;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            font-size: 0.95rem;
            color: white;
            cursor: pointer;
            transition: all 0.3s;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 1rem center;
        }

        select option {
            background: #1a1e24;
            color: white;
        }

        select:hover {
            border-color: rgba(96, 165, 250, 0.5);
            background: rgba(255, 255, 255, 0.05);
        }

        select:focus {
            outline: none;
            border-color: #60a5fa;
            box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.1);
        }

        .predict-btn {
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            color: white;
            border: none;
            padding: 1.2rem 2rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 16px;
            cursor: pointer;
            width: 100%;
            margin-top: 2rem;
            transition: transform 0.3s, box-shadow 0.3s;
            box-shadow: 0 10px 30px -10px rgba(96, 165, 250, 0.3);
            grid-column: span 2;
        }

        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 40px -10px rgba(96, 165, 250, 0.5);
        }

        /* Result Section */
        .result-section {
            margin-top: 2.5rem;
            padding: 2.5rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .result-header h2 {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .salary-display {
            text-align: center;
            margin-bottom: 2rem;
        }

        .salary-amount {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 2rem;
            line-height: 1.2;
        }

        .salary-breakdown {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }

        .breakdown-item {
            background: rgba(255, 255, 255, 0.03);
            padding: 1.5rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .breakdown-label {
            color: rgba(255, 255, 255, 0.4);
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }

        .breakdown-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: white;
        }

        /* Info Tooltip */
        .info-tooltip {
            background: rgba(96, 165, 250, 0.1);
            border-left: 4px solid #60a5fa;
            padding: 1.5rem;
            border-radius: 16px;
            margin: 2rem 0;
            color: rgba(255, 255, 255, 0.8);
        }

        .info-tooltip strong {
            color: #60a5fa;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 3rem 0;
            color: rgba(255, 255, 255, 0.3);
            font-size: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            margin-top: 3rem;
        }

        /* Loading Spinner */
        .loading {
            text-align: center;
            padding: 2rem;
        }

        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top: 4px solid #60a5fa;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Error Message */
        .error-message {
            background: rgba(220, 38, 38, 0.1);
            border-left: 4px solid #ef4444;
            color: #fecaca;
            padding: 1.5rem;
            border-radius: 16px;
            margin-top: 1.5rem;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .navbar {
                padding: 1rem 2rem;
                flex-wrap: wrap;
            }

            .nav-links {
                gap: 1.5rem;
            }

            .container {
                padding: 1rem 2rem;
            }

            .hero h1 {
                font-size: 2.5rem;
            }

            .hero-buttons {
                flex-direction: column;
                align-items: center;
            }

            .stats-container,
            .steps-grid,
            .about-content {
                grid-template-columns: 1fr;
                flex-direction: column;
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

            .form-group.full-width,
            .predict-btn {
                grid-column: span 1;
            }

            .salary-breakdown {
                grid-template-columns: 1fr;
            }

            .salary-amount {
                font-size: 3rem;
            }

            .about-stats {
                grid-template-columns: 1fr;
            }
        }

        /* Stats Banner */
        .stats-banner {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin: 2rem 0;
        }

        .stat-item {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stat-item .stat-value {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
        }

        .stat-item .stat-label {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.8rem;
        }

        @media (max-width: 768px) {
            .stats-banner {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <!-- Canvas for animated background -->
    <canvas id="canvas-background"></canvas>
    
    <!-- Glass overlay for extra depth -->
    <div class="glass-overlay"></div>

    <!-- Main Content -->
    <div class="content-wrapper">
        <nav class="navbar">
            <div class="logo">SalaryPredict</div>
            <div class="nav-links">
                <a onclick="scrollToSection('hero')" class="active">Home</a>
                <a onclick="scrollToSection('predictor')">Predictor</a>
                <a onclick="scrollToSection('how-it-works')">How It Works</a>
                <a onclick="scrollToSection('about')">About</a>
            </div>
            <div class="powered-by">
                Powered by <span>Linear Regression</span>
            </div>
        </nav>

        <main class="container">
            <!-- Hero Section -->
            <section id="hero" class="hero">
                <h1>Predict Your  <br><span>Salary with Machine LearningL</span></h1>
                <p>
                    Based on real dataset with 607 records • Multi-feature Linear Regression • 8 key factors
                </p>
                <div class="hero-buttons">
                    <button class="primary-btn" onclick="scrollToSection('predictor')">Try Predictor →</button>
                    <button class="secondary-btn" onclick="scrollToSection('how-it-works')">Learn More</button>
                </div>

                <!-- Stats Banner -->
                <div class="stats-banner">
                    <div class="stat-item">
                        <div class="stat-value">607</div>
                        <div class="stat-label">Training Samples</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">R² 0.75</div>
                        <div class="stat-label">Model Accuracy</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">8</div>
                        <div class="stat-label">Features</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">2020-27</div>
                        <div class="stat-label">Years</div>
                    </div>
                </div>
            </section>

            <!-- Predictor Section -->
            <section id="predictor" class="section">
                <h2 class="section-title">Salary Calculator</h2>
                <p class="section-subtitle">Fill in your details below to get an instant prediction</p>

                <!-- Info Tooltip -->
                <div class="info-tooltip">
                    <strong>📊 Model Features:</strong> Using 8 different factors to predict your salary: Work Year, Experience Level, Employment Type, Job Title, Location, Remote Ratio, and Company Size.
                </div>

                <div class="prediction-section">
                    <form id="salaryForm" onsubmit="event.preventDefault(); predictSalary();">
                        <div class="form-grid">
                            <!-- Work Year -->
                            <div class="form-group">
                                <label>📅 Work Year</label>
                                <select id="workYear">
                                    <option value="2020">2020</option>
                                    <option value="2021">2021</option>
                                    <option value="2022" selected>2022</option>
                                    <option value="2023">2023</option>
                                    <option value="2024">2024</option>
                                    <option value="2025">2025</option>
                                    <option value="2026">2026</option>
                                    <option value="2027">2027</option>
                                </select>
                            </div>

                            <!-- Experience Level -->
                            <div class="form-group">
                                <label>📈 Experience Level</label>
                                <select id="experienceLevel">
                                    <option value="0">EN - Entry Level</option>
                                    <option value="1">MI - Mid Level</option>
                                    <option value="2" selected>SE - Senior Level</option>
                                    <option value="3">EX - Executive Level</option>
                                </select>
                            </div>

                            <!-- Employment Type -->
                            <div class="form-group">
                                <label>💼 Employment Type</label>
                                <select id="employmentType">
                                    <option value="0" selected>FT - Full Time</option>
                                    <option value="1">PT - Part Time</option>
                                    <option value="2">CT - Contract</option>
                                    <option value="3">FL - Freelance</option>
                                </select>
                            </div>

                            <!-- Remote Ratio -->
                            <div class="form-group">
                                <label>🏠 Remote Ratio</label>
                                <select id="remoteRatio">
                                    <option value="0">0% - On-site</option>
                                    <option value="50">50% - Hybrid</option>
                                    <option value="100" selected>100% - Fully Remote</option>
                                </select>
                            </div>

                            <!-- Company Size -->
                            <div class="form-group">
                                <label>🏢 Company Size</label>
                                <select id="companySize">
                                    <option value="0">S - Small (<50 employees)</option>
                                    <option value="1" selected>M - Medium (50-250 employees)</option>
                                    <option value="2">L - Large (>250 employees)</option>
                                </select>
                            </div>

                            <!-- Company Location -->
                            <div class="form-group">
                                <label>🌍 Company Location</label>
                                <select id="companyLocation">
                                    <option value="0" selected>US - United States</option>
                                    <option value="1">GB - United Kingdom</option>
                                    <option value="2">CA - Canada</option>
                                    <option value="3">DE - Germany</option>
                                    <option value="4">FR - France</option>
                                    <option value="5">IN - India</option>
                                    <option value="6">ES - Spain</option>
                                    <option value="7">JP - Japan</option>
                                    <option value="8">BR - Brazil</option>
                                    <option value="9">AU - Australia</option>
                                    <option value="10">Other</option>
                                </select>
                            </div>

                            <!-- Job Title (Full Width) -->
                            <div class="form-group full-width">
                                <label>🔬 Job Title</label>
                                <select id="jobTitle">
                                    <option value="0">Data Scientist</option>
                                    <option value="1">Data Engineer</option>
                                    <option value="2" selected>Data Analyst</option>
                                    <option value="3">Machine Learning Engineer</option>
                                    <option value="4">Research Scientist</option>
                                    <option value="5">Data Science Manager</option>
                                    <option value="6">Big Data Engineer</option>
                                    <option value="7">Business Data Analyst</option>
                                    <option value="8">Lead Data Scientist</option>
                                    <option value="9">Principal Data Scientist</option>
                                    <option value="10">AI Scientist</option>
                                    <option value="11">Computer Vision Engineer</option>
                                    <option value="12">ML Engineer</option>
                                    <option value="13">Data Architect</option>
                                    <option value="14">Data Analytics Manager</option>
                                    <option value="15">Director of Data Science</option>
                                    <option value="16">Head of Data</option>
                                    <option value="17">Other</option>
                                </select>
                            </div>

                            <!-- Predict Button -->
                            <button type="submit" class="predict-btn">🔮 Predict My Salary</button>
                        </div>
                    </form>

                    <!-- Loading Spinner -->
                    <div class="loading" id="loading" style="display: none;">
                        <div class="spinner"></div>
                        <p style="margin-top: 10px; color: rgba(255,255,255,0.5);">Calculating...</p>
                    </div>

                    <!-- Results Section -->
                    <div id="resultSection" style="display: none;">
                        <div class="result-section">
                            <div class="result-header">
                                <h2>🎉 Your Predicted Salary</h2>
                            </div>

                            <div class="salary-display">
                                <div class="salary-amount" id="predictedSalary">$0</div>

                                <div class="salary-breakdown">
                                    <div class="breakdown-item">
                                        <div class="breakdown-label">Monthly (gross)</div>
                                        <div class="breakdown-value" id="monthlySalary">$0</div>
                                    </div>
                                    <div class="breakdown-item">
                                        <div class="breakdown-label">Daily (approx)</div>
                                        <div class="breakdown-value" id="dailySalary">$0</div>
                                    </div>
                                    <div class="breakdown-item">
                                        <div class="breakdown-label">Hourly (approx)</div>
                                        <div class="breakdown-value" id="hourlySalary">$0</div>
                                    </div>
                                </div>
                            </div>

                            <div style="margin-top: 30px; text-align: center; color: rgba(255,255,255,0.5);">
                                <p>✨ Based on your selections and historical data from 2020-2022 (extrapolated to 2027)</p>
                            </div>
                        </div>
                    </div>

                    <!-- Error Message -->
                    <div id="errorSection" style="display: none;">
                        <div class="error-message" id="errorMessage"></div>
                    </div>
                </div>
            </section>

            <!-- How It Works Section -->
            <section id="how-it-works" class="section">
                <h2 class="section-title">How It Works</h2>
                <p class="section-subtitle">Three simple steps to predict your salary</p>

                <div class="steps-grid">
                    <div class="step-card">
                        <div class="step-number">01</div>
                        <h3 class="step-title">Enter Your Details</h3>
                        <p class="step-description">
                            Fill in your work year, experience level, job title, location, and other relevant factors that influence data science salaries.
                        </p>
                    </div>

                    <div class="step-card">
                        <div class="step-number">02</div>
                        <h3 class="step-title">AI Model Analysis</h3>
                        <p class="step-description">
                            Our Linear Regression model processes your inputs using coefficients learned from 607 real-world data science salary records.
                        </p>
                    </div>

                    <div class="step-card">
                        <div class="step-number">03</div>
                        <h3 class="step-title">Get Instant Result</h3>
                        <p class="step-description">
                            Receive your predicted annual salary along with monthly, daily, and hourly breakdowns based on industry standards.
                        </p>
                    </div>
                </div>

                <div style="margin-top: 3rem; text-align: center;">
                    <div class="info-tooltip" style="display: inline-block; max-width: 600px;">
                        <strong>⚡ Formula:</strong> Salary = Intercept + Σ(Feature × Coefficient)
                    </div>
                </div>
            </section>

            <!-- About Section -->
            <section id="about" class="section">
                <h2 class="section-title">About The Project</h2>
                <p class="section-subtitle">Understanding the model behind the predictions</p>

                <div class="about-content">
                    <div class="about-text">
                        <p>
                            This Salary Predictor uses a <strong>Linear Regression model</strong> trained on the 
                            <strong>ds_salaries.csv</strong> dataset, which contains 607  salary records 
                            from 2020 to 2022 across various countries, job titles, and experience levels.
                        </p>
                        <p>
                            The model achieves an <strong>R² score of 0.75</strong>, meaning it explains 75% of the 
                            variance in salaries based on the 8 input features. For years 2023-2027, the model 
                            extrapolates based on the trend from 2020-2022.
                        </p>
                        
                        <ul class="feature-list">
                            <li>
                                <span class="feature-icon">✓</span>
                                <span><strong>8 Features:</strong> Work Year, Experience Level, Employment Type, Job Title, Location, Remote Ratio, Company Size, Employee Residence</span>
                            </li>
                            <li>
                                <span class="feature-icon">✓</span>
                                <span><strong>50+ Countries:</strong> Global dataset covering major data science markets</span>
                            </li>
                            <li>
                                <span class="feature-icon">✓</span>
                                <span><strong>80+ Job Titles:</strong> From Data Analyst to Director of Data Science</span>
                            </li>
                            <li>
                                <span class="feature-icon">✓</span>
                                <span><strong>Linear Regression:</strong> Simple but interpretable machine learning algorithm</span>
                            </li>
                        </ul>
                    </div>

                    <div class="about-stats">
                        <div class="about-stat-card">
                            <div class="about-stat-number">607</div>
                            <div class="about-stat-label">Training Records</div>
                        </div>
                        <div class="about-stat-card">
                            <div class="about-stat-number">R² 0.75</div>
                            <div class="about-stat-label">Model Accuracy</div>
                        </div>
                        <div class="about-stat-card">
                            <div class="about-stat-number">8</div>
                            <div class="about-stat-label">Features</div>
                        </div>
                        <div class="about-stat-card">
                            <div class="about-stat-number">2020-27</div>
                            <div class="about-stat-label">Prediction Years</div>
                        </div>
                    </div>
                </div>
            </section>

           <footer class="footer">
    <p>📊 Model trained on ds_salaries.csv dataset | <span style="color: #60a5fa;">Machine Learning</span> • Linear Regression</p>
    <p style="font-size: 12px; margin-top: 10px;">Note: 2023-2027 are extrapolated predictions based on 2020-2022 trends</p>
    
    <!-- Created by section -->
    <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.1);">
        <p style="color: rgba(255,255,255,0.4); font-size: 0.9rem;">
            🤖 Built with Machine Learning by 
            <span style="color: #60a5fa; font-weight: 600;">Nadia Akter Eshita</span>
        </p>
    </div>
</footer>

        </main>
    </div>

    <!-- JavaScript -->
    <script>
        // ========== ANIMATED BACKGROUND ==========
        const canvas = document.getElementById('canvas-background');
        const ctx = canvas.getContext('2d');
        
        let width, height;
        let mouseX = 0, mouseY = 0;
        let time = 0;
        
        // Blob class for floating shapes
        class Blob {
            constructor() {
                this.reset();
            }
            
            reset() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.radius = Math.random() * 200 + 100;
                this.speedX = (Math.random() - 0.5) * 0.2;
                this.speedY = (Math.random() - 0.5) * 0.2;
                this.speedFactor = Math.random() * 0.5 + 0.5;
                
                const colors = [
                    { r: 96, g: 165, b: 250 },
                    { r: 167, g: 139, b: 250 },
                    { r: 244, g: 114, b: 182 },
                    { r: 34, g: 211, b: 238 },
                    { r: 192, g: 132, b: 252 },
                ];
                const color1 = colors[Math.floor(Math.random() * colors.length)];
                const color2 = colors[Math.floor(Math.random() * colors.length)];
                
                this.color1 = `rgba(${color1.r}, ${color1.g}, ${color1.b}, 0.15)`;
                this.color2 = `rgba(${color2.r}, ${color2.g}, ${color2.b}, 0.25)`;
            }
            
            update() {
                const dx = mouseX - width/2;
                const dy = mouseY - height/2;
                const mouseInfluence = 0.0001;
                
                this.x += this.speedX * this.speedFactor + dx * mouseInfluence * this.speedFactor;
                this.y += this.speedY * this.speedFactor + dy * mouseInfluence * this.speedFactor;
                
                if (this.x < -this.radius) this.x = width + this.radius;
                if (this.x > width + this.radius) this.x = -this.radius;
                if (this.y < -this.radius) this.y = height + this.radius;
                if (this.y > height + this.radius) this.y = -this.radius;
            }
            
            draw() {
                const gradient = ctx.createRadialGradient(
                    this.x - this.radius * 0.3, 
                    this.y - this.radius * 0.3, 
                    10,
                    this.x, 
                    this.y, 
                    this.radius
                );
                gradient.addColorStop(0, this.color1);
                gradient.addColorStop(0.5, this.color2);
                gradient.addColorStop(1, 'rgba(10, 12, 16, 0)');
                
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = gradient;
                ctx.filter = `blur(${40 * this.speedFactor}px)`;
                ctx.fill();
            }
        }

        let blobs = [];
        
        function initBlobs() {
            blobs = [];
            const blobCount = Math.max(8, Math.floor(width * height / 40000));
            for (let i = 0; i < blobCount; i++) {
                blobs.push(new Blob());
            }
        }

        function animate() {
            ctx.clearRect(0, 0, width, height);
            
            const bgGradient = ctx.createRadialGradient(
                width/2, height/2, 0,
                width/2, height/2, width
            );
            bgGradient.addColorStop(0, '#0f1117');
            bgGradient.addColorStop(1, '#1a1e2a');
            ctx.fillStyle = bgGradient;
            ctx.fillRect(0, 0, width, height);
            
            ctx.filter = 'none';
            blobs.forEach(blob => {
                blob.update();
                blob.draw();
            });
            
            ctx.filter = 'blur(60px)';
            ctx.globalCompositeOperation = 'screen';
            
            const flareX = width/2 + Math.sin(time * 0.001) * 200;
            const flareY = height/2 + Math.cos(time * 0.0012) * 150;
            const flareGradient = ctx.createRadialGradient(
                flareX, flareY, 0,
                flareX, flareY, 400
            );
            flareGradient.addColorStop(0, 'rgba(96, 165, 250, 0.1)');
            flareGradient.addColorStop(1, 'rgba(167, 139, 250, 0)');
            
            ctx.beginPath();
            ctx.arc(flareX, flareY, 400, 0, Math.PI * 2);
            ctx.fillStyle = flareGradient;
            ctx.fill();
            
            ctx.globalCompositeOperation = 'source-over';
            ctx.filter = 'none';
            
            time++;
            requestAnimationFrame(animate);
        }

        function resizeCanvas() {
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width;
            canvas.height = height;
            initBlobs();
        }

        window.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = e.clientY;
        });

        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        animate();

        // ========== SCROLL FUNCTION ==========
        function scrollToSection(sectionId) {
            const section = document.getElementById(sectionId);
            if (section) {
                section.scrollIntoView({ behavior: 'smooth' });
                
                // Update active nav link
                document.querySelectorAll('.nav-links a').forEach(link => {
                    link.classList.remove('active');
                });
                event.target.classList.add('active');
            }
        }

        // Update active nav link on scroll
        window.addEventListener('scroll', () => {
            const sections = ['hero', 'predictor', 'how-it-works', 'about'];
            const scrollPosition = window.scrollY + 200;

            sections.forEach(section => {
                const sectionElement = document.getElementById(section);
                if (sectionElement) {
                    const sectionTop = sectionElement.offsetTop;
                    const sectionBottom = sectionTop + sectionElement.offsetHeight;

                    if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                        document.querySelectorAll('.nav-links a').forEach(link => {
                            link.classList.remove('active');
                            if (link.textContent.toLowerCase() === section || 
                                (section === 'hero' && link.textContent === 'Home')) {
                                link.classList.add('active');
                            }
                        });
                    }
                }
            });
        });

        // ========== MODEL PARAMETERS ==========
        const MODEL = {
            intercept: -15095881.344720656,
            coefficients: [
                7481.167570659545,    // work_year
                12531.902273660227,   // experience_level
                -14824.301873756254,  // employment_type
                1252.228415892946,    // job_title
                489.70610369674796,   // company_location
                108.99533311993673,   // remote_ratio
                -11628.470924001436,  // company_size
                0                      // employee_residence
            ]
        };

        function predictSalary() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultSection').style.display = 'none';
            document.getElementById('errorSection').style.display = 'none';

            setTimeout(() => {
                try {
                    const features = [
                        parseFloat(document.getElementById('workYear').value),
                        parseFloat(document.getElementById('experienceLevel').value),
                        parseFloat(document.getElementById('employmentType').value),
                        parseFloat(document.getElementById('jobTitle').value),
                        parseFloat(document.getElementById('companyLocation').value),
                        parseFloat(document.getElementById('remoteRatio').value),
                        parseFloat(document.getElementById('companySize').value),
                        0
                    ];

                    let salary = MODEL.intercept;
                    for (let i = 0; i < features.length; i++) {
                        salary += features[i] * MODEL.coefficients[i];
                    }
                    
                    salary = Math.max(20000, Math.min(500000, salary));

                    const monthly = salary / 12;
                    const daily = salary / 260;
                    const hourly = daily / 8;

                    document.getElementById('predictedSalary').innerHTML = 
                        '$' + Math.round(salary).toLocaleString();
                    document.getElementById('monthlySalary').innerHTML = 
                        '$' + Math.round(monthly).toLocaleString();
                    document.getElementById('dailySalary').innerHTML = 
                        '$' + Math.round(daily).toLocaleString();
                    document.getElementById('hourlySalary').innerHTML = 
                        '$' + Math.round(hourly).toLocaleString();

                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('resultSection').style.display = 'block';
                    
                    document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });

                } catch (error) {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('errorMessage').innerHTML = 'Error: ' + error.message;
                    document.getElementById('errorSection').style.display = 'block';
                }
            }, 500);
        }

        window.onload = function() {
            setTimeout(predictSalary, 100);
        };
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def home():
    """Home page - show the form"""
    return render_template_string(HTML_TEMPLATE, 
                                 prediction=None, 
                                 monthly=None,
                                 daily=None,
                                 hourly=None,
                                 loading=False,
                                 error=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and make prediction"""
    try:
        # Get all form data
        features = [
            float(request.form['work_year']),
            float(request.form['experience_level']),
            float(request.form['employment_type']),
            float(request.form['job_title']),
            float(request.form['company_location']),
            0.0,  # employee_residence (using company location as proxy)
            float(request.form['remote_ratio']),
            float(request.form['company_size'])
        ]
        
        print(f"Features received: {features}")
        
        # Calculate prediction using coefficients
        prediction = MODEL['intercept']
        for i, value in enumerate(features):
            prediction += value * MODEL['coefficients'][i]
        
        # Ensure prediction is reasonable
        prediction = max(10000, min(500000, prediction))
        
        # Calculate breakdowns
        monthly = prediction / 12
        daily = prediction / 260  # Approx working days per year
        hourly = daily / 8
        
        # Format numbers with commas
        formatted_pred = f"{prediction:,.0f}"
        formatted_monthly = f"{monthly:,.0f}"
        formatted_daily = f"{daily:,.0f}"
        formatted_hourly = f"{hourly:,.0f}"
        
        return render_template_string(
            HTML_TEMPLATE,
            prediction=formatted_pred,
            monthly=formatted_monthly,
            daily=formatted_daily,
            hourly=formatted_hourly,
            loading=False,
            error=None
        )
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template_string(
            HTML_TEMPLATE,
            prediction=None,
            monthly=None,
            daily=None,
            hourly=None,
            loading=False,
            error=f"Error making prediction: {str(e)}"
        )

@app.route('/health')
def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'model': 'coefficients_loaded',
        'years': '2020-2027'
    }

if __name__ == '__main__':
    app.run()