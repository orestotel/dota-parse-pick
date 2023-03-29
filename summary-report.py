import os
import glob
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch

def parse_data(directory):
    files = glob.glob(os.path.join(directory, '*.json'))
    data = []

    for file in files:
        with open(file, 'r') as f:
            file_data = json.load(f)
            data.extend(file_data)

    return data

def generate_summary_report(output_filename):
    data = parse_data('parse-data') + parse_data('parse-data/batches')
    num_matches = len(data)

    mmrs = np.array([match['mmr'] for match in data])
    max_mmr, min_mmr, avg_mmr = mmrs.max(), mmrs.min(), mmrs.mean()

    heroes_data = {}
    for match in data:
        radiant_win = match['radiant_win']
        for hero_id in match['picks']:
            if hero_id not in heroes_data:
                heroes_data[hero_id] = {'picks': 0, 'wins': 0}

            heroes_data[hero_id]['picks'] += 1
            if (hero_id in match['radiant_picks']) == radiant_win:
                heroes_data[hero_id]['wins'] += 1

    hero_pickrates = np.array([heroes_data[hero_id]['picks'] for hero_id in heroes_data])
    hero_winrates = np.array([heroes_data[hero_id]['wins'] / heroes_data[hero_id]['picks'] for hero_id in heroes_data])

    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    ax1.bar(heroes_data.keys(), hero_winrates, color='b', alpha=0.6)
    ax2.plot(heroes_data.keys(), hero_pickrates, 'r-')

    ax1.set_xlabel('Hero ID')
    ax1.set_ylabel('Winrate', color='b')
    ax2.set_ylabel('Pickrate', color='r')

    plt.title('Hero Winrates and Pickrates')
    plt.savefig('temp_plot.png')
    plt.clf()

    # PDF generation
    c = canvas.Canvas(output_filename, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, 10 * inch, "Matches Report")
    c.setFont("Helvetica", 12)
    c.drawString(inch, 9.5 * inch, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    c.drawString(inch, 9 * inch, "Dota-Parsepick")
    c.drawImage('temp_plot.png', inch, 4 * inch, width=6 * inch, height=4 * inch)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, 3.5 * inch, "Match Summary")
    c.setFont("Helvetica", 12)
    c.drawString(inch, 3 * inch, f"Max MMR: {max_mmr}, Min MMR: {min_mmr}, Avg MMR: {avg_mmr:.2f}")
    c.drawString(inch, 2.5 * inch, f"Total Matches: {num_matches}")
    c.drawString(inch, 2 * inch, f"Radiant Winrate: {np.mean([match['radiant_win'] for match in data]) * 100:.2f}%")
    c.drawString(inch, 1.5 * inch, f"Dire Winrate: {np.mean([not match['radiant_win'] for match in data]) * 100:.2f}%")
    c.save()

    # Cleanup
    os.remove('temp_plot.png')


if __name__ == "__main__":
    output_filename = input("Enter the name of the output PDF file: ")
    output_filename = os.path.join('parse-data', 'reports', output_filename)

    generate_summary_report(output_filename)
    print("Summary report generated successfully!")

