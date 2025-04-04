#!/usr/bin/env python3
"""
Concept Map Visualization Component for TextMind
Using streamlit-javascript to create a Cytoscape.js visualization
"""

import streamlit as st
import json
import os
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from streamlit_javascript import st_javascript

# Setup logging
logger = logging.getLogger(__name__)

def load_concept_data(concept_dict_path: str = "data/concept_dictionary.json") -> List[Dict[str, Any]]:
    """
    Load concept dictionary data from JSON file
    
    Args:
        concept_dict_path: Path to the concept dictionary JSON file
        
    Returns:
        List of concept dictionaries
    """
    try:
        # Try to load from the specified path
        logger.info(f"Attempting to load concept data from {concept_dict_path}")
        with open(concept_dict_path, 'r', encoding='utf-8') as f:
            concepts = json.load(f)
        logger.info(f"Loaded {len(concepts)} concepts from {concept_dict_path}")
        return concepts
    except Exception as primary_error:
        st.error(f"Error loading concept data from {concept_dict_path}: {str(primary_error)}")
        logger.error(f"Error loading concept data from {concept_dict_path}: {str(primary_error)}")
        
        # Fallback to CSV if JSON fails
        try:
            csv_path = concept_dict_path.replace('.json', '.csv')
            logger.info(f"Attempting to load from CSV fallback: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Convert DataFrame to list of dictionaries
            concepts = []
            for _, row in df.iterrows():
                concept_dict = {
                    "concept": row["concept"],
                    "count": row["count"],
                    "related_terms": [term.strip() for term in row["related_terms"].split(',')]
                }
                concepts.append(concept_dict)
            
            logger.info(f"Loaded {len(concepts)} concepts from CSV fallback")
            return concepts
        except Exception as csv_error:
            logger.error(f"Error loading from CSV fallback: {str(csv_error)}")
            
            # Generate minimal concept data if all else fails
            logger.info("Generating minimal concept data")
            return generate_minimal_concepts()

def generate_minimal_concepts() -> List[Dict[str, Any]]:
    """Generate minimal concept data as a last resort fallback"""
    return [
        {
            "concept": "bourgeoisie",
            "count": 79,
            "related_terms": ["bourgeois", "capitalists", "middle class", "property owners"]
        },
        {
            "concept": "proletariat",
            "count": 55,
            "related_terms": ["proletarian", "workers", "working class", "labor"]
        },
        {
            "concept": "communism",
            "count": 40,
            "related_terms": ["communist", "communistic", "socialist", "revolutionary"]
        },
        {
            "concept": "class struggle",
            "count": 25,
            "related_terms": ["class antagonism", "class conflict", "revolution", "oppression"]
        },
        {
            "concept": "capital",
            "count": 32,
            "related_terms": ["money", "property", "means of production", "wealth"]
        }
    ]

def prepare_cytoscape_data(concepts: List[Dict[str, Any]], min_count: int = 1) -> Dict[str, List[Dict[str, Any]]]:
    """
    Prepare data for Cytoscape visualization
    
    Args:
        concepts: List of concept dictionaries
        min_count: Minimum count for main concepts to include
        
    Returns:
        Dictionary containing nodes and edges for the concept map
    """
    nodes = []
    edges = []
    
    # Track processed concepts to avoid duplicates
    processed_concepts = set()
    
    # Debug info
    logger.info(f"Preparing graph data from {len(concepts)} concepts with min_count={min_count}")
    
    # Process each concept
    for concept_data in concepts:
        concept = concept_data.get("concept", "")
        if not concept:  # Skip if concept name is missing
            continue
            
        count = concept_data.get("count", 0)
        related_terms = concept_data.get("related_terms", [])
        
        # Skip concepts with count less than min_count
        if count < min_count:
            continue
            
        # Skip if already processed
        if concept in processed_concepts:
            continue
        
        # Add main concept node
        size = min(60, max(30, 20 + count / 2))  # Scale node size based on count
        
        nodes.append({
            "data": {
                "id": concept,
                "label": concept,
                "size": size,
                "count": count,
                "type": "main_concept"
            }
        })
        
        processed_concepts.add(concept)
        
        # Add related terms and edges
        for term in related_terms:
            if not term:  # Skip empty terms
                continue
                
            # Skip if already processed
            if term in processed_concepts:
                # Just add edge if the term exists
                edges.append({
                    "data": {
                        "source": concept,
                        "target": term,
                        "weight": 1,
                        "type": "related"
                    }
                })
                continue
            
            # Add node for related term
            nodes.append({
                "data": {
                    "id": term,
                    "label": term,
                    "size": 20,  # Smaller size for related terms
                    "count": 0,
                    "type": "related_term"
                }
            })
            
            processed_concepts.add(term)
            
            # Add edge
            edges.append({
                "data": {
                    "source": concept,
                    "target": term,
                    "weight": 1,
                    "type": "related"
                }
            })
    
    # Add a couple of core concepts if we don't have any nodes yet
    if not nodes and concepts:
        # Just add a few core concepts directly
        base_concepts = ["bourgeoisie", "proletariat", "communism", "capitalism", "revolution"]
        for concept in base_concepts:
            nodes.append({
                "data": {
                    "id": concept,
                    "label": concept,
                    "size": 40,
                    "count": 10,
                    "type": "main_concept"
                }
            })
            
            # Add some basic connections
            if len(nodes) > 1:
                prev_concept = nodes[-2]["data"]["id"]
                edges.append({
                    "data": {
                        "source": concept,
                        "target": prev_concept,
                        "weight": 1,
                        "type": "concept_concept"
                    }
                })
    
    logger.info(f"Prepared {len(nodes)} nodes and {len(edges)} edges for visualization")
    return {
        "nodes": nodes,
        "edges": edges
    }

def get_concept_info(concept: str, concepts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific concept
    
    Args:
        concept: Concept name to look up
        concepts: List of concept dictionaries
        
    Returns:
        Dictionary with concept information or None if not found
    """
    for concept_data in concepts:
        if concept_data.get("concept", "") == concept:
            return concept_data
    
    # If not found as a main concept, check in related terms
    for concept_data in concepts:
        related_terms = concept_data.get("related_terms", [])
        if concept in related_terms:
            return {
                "concept": concept,
                "related_to": concept_data.get("concept", ""),
                "count": 0,
                "related_terms": [],
                "is_related_term": True
            }
    
    return None

def render_fallback_visualization(concepts: List[Dict[str, Any]], min_count: int = 1):
    """
    Render a simple visualization of concepts using native Streamlit components
    as a fallback if Cytoscape.js fails to load
    
    Args:
        concepts: List of concept dictionaries
        min_count: Minimum frequency to filter concepts
    """
    st.subheader("Key Marxist Concepts (Fallback Visualization)")
    
    # Filter concepts by min_count
    filtered_concepts = [c for c in concepts if c.get("count", 0) >= min_count]
    
    if not filtered_concepts:
        st.warning(f"No concepts found with frequency >= {min_count}. Showing all concepts.")
        filtered_concepts = concepts
    
    # Sort concepts by count (frequency)
    filtered_concepts.sort(key=lambda x: x.get("count", 0), reverse=True)
    
    # Display concept count
    st.info(f"Showing {len(filtered_concepts)} concepts")
    
    # Display data in an expander for debugging
    with st.expander("Debug: Concept Data", expanded=False):
        st.write("Raw concept data format:", type(concepts))
        
        if concepts and isinstance(concepts, list) and len(concepts) > 0:
            st.write("First concept structure:", list(concepts[0].keys()))
            st.write("Sample concept:", concepts[0])
        else:
            st.write("Concept data is empty or invalid")
    
    # Create columns
    cols = st.columns(3)
    
    # Display concepts in grid
    for i, concept_data in enumerate(filtered_concepts):
        concept = concept_data.get("concept", f"Concept {i+1}")
        count = concept_data.get("count", 0)
        col_idx = i % 3
        
        with cols[col_idx]:
            with st.expander(f"**{concept.title()}** ({count})", expanded=False):
                # Display related terms
                related_terms = concept_data.get("related_terms", [])
                if related_terms:
                    st.markdown("**Related terms:**")
                    st.write(", ".join(related_terms))
                
                # Display sample context if available
                if "sample_contexts" in concept_data and concept_data.get("sample_contexts", []):
                    contexts = concept_data.get("sample_contexts", [])
                    if contexts:
                        context = contexts[0]
                        st.markdown("**Sample context:**")
                        section = context.get("section", "Unknown section")
                        st.markdown(f"*From: {section}*")
                        
                        # Format text with highlighted term
                        text = context.get("text", "")
                        matched_term = context.get("matched_term", concept)
                        snippet = text[:200] + "..." if len(text) > 200 else text
                        if matched_term in snippet:
                            highlighted = snippet.replace(matched_term, f"**{matched_term}**")
                            st.markdown(highlighted)
                        else:
                            st.markdown(snippet)

def render_concept_map():
    """
    Render the concept map UI component using Cytoscape.js via streamlit-javascript
    """
    st.title("Marxist Concept Map")
    st.write("Explore the relationships between key concepts in Marx's Communist Manifesto")
    
    # Add a direct debugging section
    debug_container = st.empty()
    
    # Load concept data
    concepts = load_concept_data()
    
    # Debug stats
    with debug_container.container():
        if not concepts:
            st.error("No concept data available. Please check the data files.")
            return
        else:
            st.success(f"Loaded {len(concepts)} concepts successfully")
    
    # Log concept data for debugging
    st.session_state.concepts_loaded = True
    st.session_state.num_concepts = len(concepts)
    logger.info(f"Concept map: loaded {len(concepts)} concepts")
    
    # Add filters in sidebar
    st.sidebar.title("Concept Map Controls")
    
    # Filter by minimum count
    min_count = st.sidebar.slider(
        "Minimum concept frequency", 
        min_value=1, 
        max_value=30, 
        value=1,
        help="Filter concepts by minimum number of occurrences in the text"
    )
    
    # Options for visualization type
    viz_type = st.sidebar.radio(
        "Visualization Type",
        ["Simple Table", "Interactive Graph", "Fallback Grid"],
        index=0,
        help="Select the visualization type for concepts"
    )
    
    # Show a simple table of concepts immediately
    if viz_type == "Simple Table":
        display_concept_table(concepts, min_count)
        return
        
    # Show the fallback visualization
    if viz_type == "Fallback Grid":
        render_fallback_visualization(concepts, min_count)
        return
    
    # Options for layout if using interactive graph
    layout_options = {
        "Concentric": "concentric", 
        "Cose": "cose",
        "Circle": "circle",
        "Grid": "grid",
        "Breadthfirst": "breadthfirst"
    }
    
    selected_layout = st.sidebar.selectbox("Layout", options=list(layout_options.keys()), index=0)
    
    # Prepare data for visualization
    cytoscape_data = prepare_cytoscape_data(concepts, min_count=min_count)
    
    # Show concept stats
    main_concepts = [n for n in cytoscape_data["nodes"] if n["data"]["type"] == "main_concept"]
    related_terms = [n for n in cytoscape_data["nodes"] if n["data"]["type"] == "related_term"]
    
    st.sidebar.markdown(f"**Showing:** {len(main_concepts)} main concepts, {len(related_terms)} related terms, {len(cytoscape_data['edges'])} connections")
    
    # Show warning if no concepts match the filter
    if not cytoscape_data["nodes"]:
        st.warning(f"No concepts found with frequency >= {min_count}. Try lowering the minimum frequency.")
        render_fallback_visualization(concepts, min_count=1)  # Show fallback with min_count 1
        return
    
    # Create div for the cytoscape visualization
    cy_container = st.empty()
    cy_container.markdown('<div id="cy" style="width: 100%; height: 600px; border: 1px solid #ccc; border-radius: 5px;"></div>', unsafe_allow_html=True)
    
    # Add a container for displaying concept info
    info_container = st.container()
    
    # Debug view of the graph data
    with st.expander("Debug: Graph Data", expanded=False):
        st.write(f"Nodes: {len(cytoscape_data['nodes'])}")
        st.write(f"Edges: {len(cytoscape_data['edges'])}")
        if cytoscape_data['nodes']:
            st.write("Sample node:", cytoscape_data['nodes'][0])
        if cytoscape_data['edges']:
            st.write("Sample edge:", cytoscape_data['edges'][0])
    
    # Create the JavaScript for Cytoscape
    cytoscape_js = f"""
    // Function to load script and check when it's loaded
    function loadScript(url, callback) {{
        const script = document.createElement('script');
        script.src = url;
        script.onload = callback;
        script.onerror = function() {{
            console.error('Failed to load script: ' + url);
            window.Streamlit.setComponentValue("script_load_failed:" + url);
        }};
        document.head.appendChild(script);
    }}

    // Function to check if jQuery is loaded 
    function loadDependencies() {{
        if (typeof jQuery === 'undefined') {{
            loadScript('https://code.jquery.com/jquery-3.6.0.min.js', loadCytoscape);
        }} else {{
            loadCytoscape();
        }}
    }}

    // Function to load Cytoscape.js if needed
    function loadCytoscape() {{
        if (typeof cytoscape === 'undefined') {{
            loadScript('https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js', initCytoscape);
        }} else {{
            initCytoscape();
        }}
    }}

    // Initialize cytoscape graph
    function initCytoscape() {{
        try {{
            if (typeof cytoscape === 'undefined') {{
                console.error('Cytoscape.js failed to load');
                window.Streamlit.setComponentValue("cytoscape_not_loaded");
                return;
            }}
            
            if (!document.getElementById('cy')) {{
                console.error('Cy container not found');
                window.Streamlit.setComponentValue("container_not_found");
                return;
            }}
            
            // Parse graph data
            const graphData = {json.dumps(cytoscape_data)};
            const layout = '{layout_options[selected_layout]}';
            
            if (!graphData || !graphData.nodes || !graphData.edges) {{
                console.error('Invalid graph data', graphData);
                window.Streamlit.setComponentValue("invalid_graph_data");
                return;
            }}
            
            // Log what we're about to render
            console.log("Initializing Cytoscape with", graphData.nodes.length, "nodes and", 
                        graphData.edges.length, "edges using", layout, "layout");
            
            // Initialize Cytoscape graph
            const cy = cytoscape({{
                container: document.getElementById('cy'),
                elements: [...graphData.nodes, ...graphData.edges],
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'label': 'data(label)',
                            'width': 'data(size)',
                            'height': 'data(size)',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'color': '#ffffff',
                            'text-outline-width': 2,
                            'text-outline-color': '#1f77b4',
                            'background-color': '#1f77b4',
                            'font-size': '12px'
                        }}
                    }},
                    {{
                        selector: 'node[type = "main_concept"]',
                        style: {{
                            'background-color': '#ff7f0e',
                            'text-outline-color': '#ff7f0e',
                            'font-weight': 'bold',
                            'font-size': '14px'
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 1,
                            'line-color': '#999999',
                            'curve-style': 'bezier'
                        }}
                    }},
                    {{
                        selector: 'edge[type = "concept_concept"]',
                        style: {{
                            'line-style': 'dashed',
                            'line-color': '#ff7f0e',
                            'width': 2
                        }}
                    }}
                ],
                layout: {{
                    name: layout,
                    padding: 30,
                    fit: true,
                    animate: true,
                    nodeDimensionsIncludeLabels: true,
                    idealEdgeLength: 100,
                    nodeSpacing: 50,
                    randomize: true
                }}
            }});
            
            // Add click event for nodes
            cy.on('tap', 'node', function(evt) {{
                const node = evt.target;
                const nodeId = node.id();
                
                // Send the selected node ID to Python
                window.Streamlit.setComponentValue(nodeId);
            }});
            
            // Make the graph responsive
            window.addEventListener('resize', function() {{
                cy.resize();
                cy.fit();
            }});
            
            // Initial fit and notify success
            cy.fit();
            window.Streamlit.setComponentValue("graph_initialized:" + graphData.nodes.length + ":" + graphData.edges.length);
        }} catch (error) {{
            console.error('Error initializing Cytoscape:', error);
            window.Streamlit.setComponentValue("error:" + error.message);
        }}
    }}

    // Start the process
    loadDependencies();
    """
    
    # Run the Cytoscape JavaScript
    selected_node = st_javascript(cytoscape_js)
    
    # Show a loading indicator while waiting for the graph to initialize
    if not selected_node:
        with st.spinner("Initializing concept map..."):
            st.info("Loading concept visualization... If this takes too long, try the fallback visualization below.")
            
            # Add a button to toggle fallback visualization
            if st.button("Show Fallback Visualization"):
                render_fallback_visualization(concepts, min_count)
                return
    
    # Check for error states in the response
    if selected_node and selected_node.startswith("error:"):
        st.error(f"Error rendering concept map: {selected_node.replace('error:', '')}")
        st.info("Showing fallback visualization instead:")
        render_fallback_visualization(concepts, min_count)
        return
    elif selected_node and selected_node.startswith("script_load_failed:"):
        st.error(f"Failed to load required scripts: {selected_node.replace('script_load_failed:', '')}")
        st.info("Showing fallback visualization instead:")
        render_fallback_visualization(concepts, min_count)
        return
    elif selected_node and selected_node == "cytoscape_not_loaded":
        st.error("Failed to load Cytoscape.js library")
        st.info("Showing fallback visualization instead:")
        render_fallback_visualization(concepts, min_count)
        return
    elif selected_node and selected_node == "container_not_found":
        st.error("Visualization container not found on page")
        st.info("Showing fallback visualization instead:")
        render_fallback_visualization(concepts, min_count)
        return
    elif selected_node and selected_node == "invalid_graph_data":
        st.error("Invalid graph data - could not render concept map")
        st.info("Showing fallback visualization instead:")
        render_fallback_visualization(concepts, min_count)
        return
    elif selected_node and selected_node.startswith("graph_initialized:"):
        # Graph initialized successfully
        parts = selected_node.split(":")
        if len(parts) > 2:
            st.success(f"Visualization ready with {parts[1]} concepts and {parts[2]} connections")
    elif selected_node and not selected_node.startswith("graph_initialized:"):
        # Display information about the selected concept
        concept_info = get_concept_info(selected_node, concepts)
        
        if concept_info:
            with info_container:
                st.markdown("---")
                st.subheader(f"Concept: {selected_node}")
                
                if concept_info.get("is_related_term"):
                    st.write(f"Related to main concept: **{concept_info.get('related_to', '')}**")
                else:
                    # Display frequency
                    st.write(f"Frequency in text: **{concept_info.get('count', 0)} occurrences**")
                    
                    # Display related terms
                    related_terms = concept_info.get("related_terms", [])
                    if related_terms:
                        st.write("**Related terms:**")
                        st.write(", ".join(related_terms))
                    
                    # Display sample contexts if available
                    sample_contexts = concept_info.get("sample_contexts", [])
                    if sample_contexts:
                        with st.expander("Sample contexts from the text", expanded=True):
                            context = sample_contexts[0]
                            st.write(f"**From section:** {context.get('section', 'Unknown')}")
                            
                            # Format the text to highlight the matched term
                            text = context.get("text", "")
                            matched_term = context.get("matched_term", "")
                            
                            # Simple highlighting for the matched term
                            if matched_term and matched_term in text:
                                highlighted_text = text.replace(
                                    matched_term, 
                                    f"**{matched_term}**"
                                )
                                st.write(highlighted_text)
                            else:
                                st.write(text)
    
    # Add some instructions
    st.markdown("---")
    st.caption("Click on a concept to see details. Adjust the minimum frequency and layout using the sidebar controls.")

def display_concept_table(concepts: List[Dict[str, Any]], min_count: int = 1):
    """
    Display concepts in a simple table format
    
    Args:
        concepts: List of concept dictionaries
        min_count: Minimum frequency to filter concepts
    """
    # Filter concepts by min_count
    filtered_concepts = [c for c in concepts if c.get("count", 0) >= min_count]
    
    if not filtered_concepts:
        st.warning(f"No concepts found with frequency >= {min_count}. Try lowering the minimum frequency.")
        filtered_concepts = concepts
    
    # Sort concepts by count (frequency)
    filtered_concepts.sort(key=lambda x: x.get("count", 0), reverse=True)
    
    # Create a DataFrame for display
    concept_data = []
    for concept in filtered_concepts:
        concept_data.append({
            "Concept": concept.get("concept", ""),
            "Frequency": concept.get("count", 0),
            "Related Terms": ", ".join(concept.get("related_terms", []))
        })
    
    # Display the data as a table
    df = pd.DataFrame(concept_data)
    st.subheader(f"Key Marxist Concepts (Total: {len(filtered_concepts)})")
    st.dataframe(df, use_container_width=True)
    
    # Allow selecting a concept to see details
    selected_concept = st.selectbox("Select a concept to see details:", 
                                   [c.get("concept", "") for c in filtered_concepts],
                                   index=0 if filtered_concepts else None)
    
    if selected_concept:
        display_concept_details(selected_concept, concepts)

def display_concept_details(concept_name: str, concepts: List[Dict[str, Any]]):
    """
    Display detailed information about a selected concept
    
    Args:
        concept_name: Name of the concept to display
        concepts: List of all concept dictionaries
    """
    concept_info = get_concept_info(concept_name, concepts)
    
    if not concept_info:
        st.warning(f"No information found for concept: {concept_name}")
        return
    
    st.markdown("---")
    st.subheader(f"Concept Details: {concept_name}")
    
    # Display basic info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Frequency", concept_info.get("count", 0))
    
    # Display related terms
    related_terms = concept_info.get("related_terms", [])
    if related_terms:
        st.subheader("Related Terms")
        st.write(", ".join(related_terms))
    
    # Display sample contexts
    sample_contexts = concept_info.get("sample_contexts", [])
    if sample_contexts:
        st.subheader("Sample Contexts")
        for i, context in enumerate(sample_contexts[:3]):  # Show up to 3 contexts
            with st.expander(f"Context #{i+1} - From: {context.get('section', 'Unknown section')}", expanded=i==0):
                text = context.get("text", "")
                matched_term = context.get("matched_term", concept_name)
                
                # Highlight the matched term
                if matched_term and matched_term in text:
                    highlighted_text = text.replace(matched_term, f"**{matched_term}**")
                    st.write(highlighted_text)
                else:
                    st.write(text)

if __name__ == "__main__":
    render_concept_map() 