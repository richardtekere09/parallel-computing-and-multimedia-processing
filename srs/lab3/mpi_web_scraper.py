from mpi4py import MPI
import requests
from bs4 import BeautifulSoup
import time
import os
import matplotlib.pyplot as plt

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

output_dir = "/Users/richard/parallel-computing-and-multimedia-processing/output/lab3"
os.makedirs(output_dir, exist_ok=True)


def get_province_links():
    url = "https://en.wikipedia.org/wiki/Provinces_of_Indonesia"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    # Locate the table or list containing the Indonesian provinces
    # For this Wikipedia page, let's focus on the table of provinces
    table = soup.find("table", {"class": "wikitable"})
    if table:
        for row in table.find_all("tr")[1:]:  # Skip header row
            cell = row.find("td")
            if cell:
                a = cell.find("a", href=True)
                if a:
                    full_url = "https://en.wikipedia.org" + a["href"]
                    links.append(full_url)
    else:
        print("No wikitable found!")
        print(f"Total province links found: {len(links)}")
    return list(set(links))


def analyze_province(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    if content is None:
        print(f"No content found in {url}")
        return (url, {})

    # Initialize result dictionary
    result = {}

    # Find all sections (headings)
    sections = content.find_all(["h2", "h3"])
    for i, section in enumerate(sections):
        section_title = section.get_text().strip()
        next_node = section.find_next_sibling()
        section_content = []

        # Collect content until the next heading
        while next_node and next_node.name not in ["h2", "h3"]:
            section_content.append(next_node)
            next_node = next_node.find_next_sibling()

        # Create a BeautifulSoup object containing just this section's content
        section_soup = BeautifulSoup(
            "".join(str(tag) for tag in section_content), "html.parser"
        )

        # Count images, tables, and references
        images = len(section_soup.find_all("img"))
        tables = len(section_soup.find_all("table"))
        refs = len(section_soup.find_all("a", href=True))

        result[section_title] = {"I": images, "T": tables, "R": refs}
        print(f"Section: {section_title}, I: {images}, T: {tables}, R: {refs}")

    return (url, result)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})

    # Ignore sidebar and footer
    for unwanted in content.select(
        ".mw-parser-output .hatnote, .reflist, #See_also, #Notes"
    ):
        unwanted.decompose()

    sections = content.find_all(["h2", "h3"])
    result = {}

    current_section = None
    for elem in content.children:
        if elem.name in ["h2", "h3"]:
            current_section = elem.get_text().strip()
            print(f"Found section: {current_section}")
            result[current_section] = {"I": 0, "T": 0, "R": 0}
        elif current_section:
            images = len(elem.find_all("img")) if hasattr(elem, "find_all") else 0
            tables = len(elem.find_all("table")) if hasattr(elem, "find_all") else 0
            refs = (
                len(elem.find_all("a", href=True)) if hasattr(elem, "find_all") else 0
            )
            result[current_section]["I"] += images
            result[current_section]["T"] += tables
            result[current_section]["R"] += refs

    return (url, result)


if rank == 0:
    start_time = time.time()
    province_links = get_province_links()
    chunks = [province_links[i::size] for i in range(size)]
else:
    chunks = None

my_links = comm.scatter(chunks, root=0)

my_results = []
for link in my_links:
    try:
        result = analyze_province(link)
        my_results.append(result)
    except Exception as e:
        print(f"Error processing {link}: {e}")

# Save individual process results
result_file = os.path.join(output_dir, f"process_{rank}_results.txt")
with open(result_file, "w", encoding="utf-8") as f:
    for url, section_data in my_results:
        f.write(f"URL: {url}\n")
        for section, counts in section_data.items():
            f.write(
                f"Section: {section}, I: {counts['I']}, T: {counts['T']}, R: {counts['R']}\n"
            )
        f.write("\n")

# Gather and process results
all_results = comm.gather(my_results, root=0)

if rank == 0:
    combined = {}
    for proc_results in all_results:
        for url, section_data in proc_results:
            for section, counts in section_data.items():
                if section not in combined:
                    combined[section] = []
                combined[section].append((url, counts))

    summary_file = os.path.join(output_dir, "final_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        for section, data in combined.items():
            max_sum = 0
            max_province = None
            for url, counts in data:
                total = counts["I"] + counts["T"] + counts["R"]
                if total > max_sum:
                    max_sum = total
                    max_province = url
            f.write(
                f"Section: {section}, Max Province: {max_province}, Total I+T+R: {max_sum}\n"
            )

    end_time = time.time()
    execution_time = end_time - start_time

    # Save execution time
    time_file = os.path.join(output_dir, f"execution_time_{size}_processes.txt")
    with open(time_file, "w") as f:
        f.write(f"Execution Time with {size} processes: {execution_time:.2f} seconds\n")

    print(f"\nExecution Time with {size} processes: {execution_time:.2f} seconds")

    # Plot execution time graph if multiple tests are done
    times = []
    processes = [2, 3, 6]
    for p in processes:
        time_path = os.path.join(output_dir, f"execution_time_{p}_processes.txt")
        if os.path.exists(time_path):
            with open(time_path) as f:
                line = f.readline().split(":")[-1].strip().split()[0]
                times.append(float(line))

    if len(times) == len(processes):
        plt.figure(figsize=(8, 6))
        plt.bar([str(p) for p in processes], times)
        plt.xlabel("Number of Processes")
        plt.ylabel("Execution Time (seconds)")
        plt.title("Execution Time vs Number of Processes")
        plt.savefig(os.path.join(output_dir, "execution_time_plot.png"))
        plt.show()
