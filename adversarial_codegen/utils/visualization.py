import os
import textwrap

import cairosvg


def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_pass_rates(results):
    # Keeping existing function unchanged
    pass_rates = {'base': 0, 'plus': 0}
    total_tasks = 0
    failed_tasks = {'base': set(), 'plus': set()}
    
    eval_data = results['eval']
    for task_name, task_results in eval_data.items():
        for result in task_results:
            task_id = result['task_id']
            if result['base_status'] != 'pass':
                failed_tasks['base'].add(task_id)
            if result['plus_status'] != 'pass':
                failed_tasks['plus'].add(task_id)
            total_tasks += 1
    
    pass_rates['base'] = (total_tasks - len(failed_tasks['base'])) / total_tasks * 100
    pass_rates['plus'] = (total_tasks - len(failed_tasks['plus'])) / total_tasks * 100
    
    return pass_rates, failed_tasks, total_tasks


def wrap_text(text, width=70):
    """Wrap long lists of elements into multiple lines"""
    return '\n'.join(textwrap.wrap(text, width=width))


def calculate_text_height(elements, width=60):
    """Calculate the height needed for a set of elements"""
    wrapped_text = wrap_text(', '.join(map(str, sorted(elements))), width)
    num_lines = len(wrapped_text.split('\n'))
    return num_lines * 20 + 40  # 20px per line + 40px padding


def svg_to_png(svg_string, output_path):
    # Add white background rectangle to SVG
    if '<svg' in svg_string:
        # Insert background rectangle right after the svg tag
        svg_parts = svg_string.split('<svg', 1)
        svg_with_bg = (f'{svg_parts[0]}<svg{svg_parts[1].split(">", 1)[0]}>'
                      f'<rect width="100%" height="100%" fill="white"/>'
                      f'{svg_parts[1].split(">", 1)[1]}')
    else:
        svg_with_bg = svg_string
    
    # Convert to PNG
    cairosvg.svg2png(bytestring=svg_with_bg, write_to=output_path)


def create_venn_diagram(set_a, set_b, exp_name, output_folder='venn', dataset_type='base'):
    # Calculate the different regions
    only_a = set_a - set_b
    only_b = set_b - set_a
    intersection = set_a & set_b
    
    # Calculate required heights for each section
    height_a = calculate_text_height(only_a)
    height_b = calculate_text_height(only_b)
    height_intersection = calculate_text_height(intersection) - 60
    
    # Add spacing between sections (50px each)
    total_legend_height = height_a + height_b + height_intersection + 200
    
    # Ensure minimum height of 400px and maximum of 2000px
    total_legend_height = max(400, min(2000, total_legend_height))
    
    # Calculate SVG dimensions
    svg_height = max(1000, total_legend_height + 100)
    
    # Format the elements for each region
    only_a_text = wrap_text(', '.join(map(str, sorted(only_a))))
    only_b_text = wrap_text(', '.join(map(str, sorted(only_b))))
    intersection_text = wrap_text(', '.join(map(str, sorted(intersection))))
    
    # Calculate Y positions for sections
    y_pos_a = 80
    y_pos_b = y_pos_a + height_a + 50
    y_pos_intersection = y_pos_b + height_b + 50
    
    # Create the dynamic legend content
    legend_a = '\n'.join(
        f'<text x="40" y="{y_pos_a + 40 + i*20}" class="legend-text">{line}</text>' 
        for i, line in enumerate(only_a_text.split('\n'))
    )
    legend_b = '\n'.join(
        f'<text x="40" y="{y_pos_b + 40 + i*20}" class="legend-text">{line}</text>'
        for i, line in enumerate(only_b_text.split('\n'))
    )
    legend_intersection = '\n'.join(
        f'<text x="40" y="{y_pos_intersection + 40 + i*20}" class="legend-text">{line}</text>'
        for i, line in enumerate(intersection_text.split('\n'))
    )
    
    # Create the SVG content with dynamic heights
    svg_content = f'''
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 {svg_height}">
        <defs>
            <style>
                .circle {{ fill-opacity: 0.3; stroke: black; stroke-width: 2; }}
                .set-label {{ font-size: 24px; font-weight: bold; }}
                .region-label {{ font-size: 20px; }}
                .legend-text {{ font-size: 16px; }}
                .legend-title {{ font-size: 20px; font-weight: bold; }}
                .section-title {{ font-size: 18px; font-weight: bold; }}
                .main-title {{ font-size: 32px; font-weight: bold; text-anchor: middle; }}
            </style>
        </defs>

        <!-- Main Title -->
        <text x="400" y="50" class="main-title">{exp_name + ' on ' + dataset_type}</text>
        
        <!-- Venn Diagram Group (left side) -->
        <g transform="translate(50, 100)">
            <circle cx="350" cy="300" r="150" class="circle" fill="#FF6B6B"/>
            <text x="230" y="250" class="set-label">Set A</text>
            
            <circle cx="450" cy="300" r="150" class="circle" fill="#4ECDC4"/>
            <text x="500" y="250" class="set-label">Set B</text>
            
            <text x="230" y="320" class="region-label">A - B</text>
            <text x="510" y="320" class="region-label">B - A</text>
            <text x="370" y="320" class="region-label">A ∩ B</text>
        </g>

        <!-- Legend Group (right side) -->
        <g transform="translate(800, 50)">
            <rect x="0" y="0" width="700" height="{total_legend_height}" fill="#fff" stroke="#000" stroke-width="1"/>
            <text x="20" y="40" class="legend-title">Set Elements:</text>
            
            <!-- Section A -->
            <text x="20" y="{y_pos_a}" class="section-title">A - B (Original Model): {len(only_a)}</text>
            <rect x="20" y="{y_pos_a + 10}" width="660" height="{height_a}" fill="#FFF" stroke="#DDD"/>
            {legend_a}
            
            <!-- Section B -->
            <text x="20" y="{y_pos_b}" class="section-title">B - A (Adversarial Model): {len(only_b)}</text>
            <rect x="20" y="{y_pos_b + 10}" width="660" height="{height_b}" fill="#FFF" stroke="#DDD"/>
            {legend_b}
            
            <!-- Intersection -->
            <text x="20" y="{y_pos_intersection}" class="section-title">A ∩ B (Intersection): {len(intersection)}</text>
            <rect x="20" y="{y_pos_intersection + 10}" width="660" height="{height_intersection}"
                fill="#FFF" stroke="#DDD"/>
            {legend_intersection}
        </g>
    </svg>
    '''
    
    svg_to_png(svg_content, f"{output_folder}/venn_{dataset_type}.png")


def visualizer(original_results, adversarial_results, exp_name, results_folder):
    # Calculate pass rates and get failed tasks
    original_rates, original_failed, total_tasks = calculate_pass_rates(original_results)
    adversarial_rates, adversarial_failed, _ = calculate_pass_rates(adversarial_results)

    os.makedirs(results_folder, exist_ok=True)
    
    # Open file in append mode ('a') or write mode ('w')
    with open(f"{results_folder}/results", 'a') as f:
        # Write and print each line
        output = [
            f"\nResults for experiment: {exp_name}",
            "\nPass Rates for Original Model:",
            f"Base Dataset: {original_rates['base']:.2f}%",
            f"Plus Dataset: {original_rates['plus']:.2f}%",
            "\nPass Rates for Adversarial Model:",
            f"Base Dataset: {adversarial_rates['base']:.2f}%",
            f"Plus Dataset: {adversarial_rates['plus']:.2f}%"
        ]
        
        for line in output:
            print(line)  # Print to console
            f.write(line + '\n')  # Write to file
    
    # Create Venn diagrams
    create_venn_diagram(original_failed['base'], adversarial_failed['base'], exp_name, results_folder, 'base')
    create_venn_diagram(original_failed['plus'], adversarial_failed['plus'], exp_name, results_folder, 'plus')

    print(f"The visualization results have been saved to {results_folder}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    import json
    with open(f"{path}/adversarial_results.json", 'r') as f:
        adversarial_results = json.load(f)
    with open(f"{path}/original_results.json", 'r') as f:
        original_results = json.load(f)
    visualizer(original_results, adversarial_results, path.split('/')[-1], path)
