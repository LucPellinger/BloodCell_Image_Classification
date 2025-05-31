# app/components/custom_css.py

# ======= Custom CSS for Gradio App (Light Mode Only) =======

custom_css = """


/* ======= Card-Like Content Blocks ======= */
.gr-block, .gr-box, .gr-column, .gr-row, .gr-group, 
.gr-plot, .gr-markdown, .gr-html, .gr-image, 
.gr-textbox, .gr-button, .gr-dataframe {
    background-color: #ffffff;
    color: #000000;
    border-radius: 12px;
    padding: 0.75rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    transition: background-color 0.3s ease-in-out, color 0.3s ease-in-out;
}

/* ======= Header Component Styles ======= */
h1, h2, h3, h4 {
    border-bottom: 0;
    margin: 0;
    padding: 0;
}

.header-container {
    display: flex;
    flex-direction: column;
    gap: 0.5em;
    padding-bottom: 0.5em;
}

.header-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.header-title {
    font-size: 1.8em;
    font-weight: bold;
}

.header-description {
    font-size: 1.1em;
    color: #333333;
}

.header-logo {
    height: 80px;
    border-radius: 10px;
}
"""
