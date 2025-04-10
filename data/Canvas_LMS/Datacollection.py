import requests
from bs4 import BeautifulSoup
import os

# === CONFIG ===
ACCESS_TOKEN = ''
BASE_URL = 'https://canvas.instructure.com/api/v1'
HEADERS = {'Authorization': f'Bearer {ACCESS_TOKEN}'}
DOWNLOAD_DIR = 'canvas_files'
ALLOWED_EXTENSIONS = ('.pdf', '.csv', '.ppt', '.pptx')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# === STEP 1: GET COURSES ===
def get_all_courses():
    url = f"{BASE_URL}/courses?per_page=100&enrollment_state=active"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def select_course_by_name(courses, search_term):
    matches = [c for c in courses if search_term.lower() in c['name'].lower()]
    if not matches:
        print("❌ No matching course found.")
        return None
    elif len(matches) == 1:
        return matches[0]
    else:
        print("🔍 Multiple matches found:")
        for i, c in enumerate(matches):
            print(f"{i+1}. {c['name']} (ID: {c['id']})")
        choice = int(input("Select course number: ")) - 1
        return matches[choice]
    
# === STEP 2: MODULE HELPERS ===
def get_modules(course_id):
    url = f"{BASE_URL}/courses/{course_id}/modules?per_page=100"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def get_module_items(course_id, module_id):
    url = f"{BASE_URL}/courses/{course_id}/modules/{module_id}/items?include[]=content_details&per_page=100"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def get_page_body(course_id, page_url):
    url = f"{BASE_URL}/courses/{course_id}/pages/{page_url}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json().get('body', '')


# === DOWNLOAD UTILITIES ===
def download_file(file_url, filename):
    print(f"⬇️  Downloading: {filename}")
    response = requests.get(file_url, headers=HEADERS)
    if response.status_code == 200:
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"✅ Saved: {filepath}")
    else:
        print(f"❌ Failed to download {filename} (Status {response.status_code})")

def extract_files_from_page_html(course_id, page_url):
    html = get_page_body(course_id, page_url)
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a', href=True):
        href = link['href']
        if any(href.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            filename = href.split('/')[-1]
            download_file(href, filename)

# === MAIN WORKFLOW ===
def download_all_files_from_course(course_id):
    modules = get_modules(course_id)
    print(f"📚 Found {len(modules)} modules.")
    for module in modules:
        print(f"\n📦 Module: {module['name']}")
        items = get_module_items(course_id, module['id'])
        for item in items:
            if item['type'] == 'Page':
                page_url = item.get('page_url')
                if page_url:
                    print(f"🔍 Scanning page: {item['title']}")
                    extract_files_from_page_html(course_id, page_url)
            elif item['type'] == 'File':
                details = item.get('content_details', {})
                filename = details.get('display_name')
                file_url = details.get('url')
                if filename and file_url and filename.endswith(ALLOWED_EXTENSIONS):
                    download_file(file_url, filename)

# === RUN ===
if __name__ == '__main__':
    print("Fetching your courses from cnavas")
    courses = get_all_courses()
    for c in courses:
        print(f"Here are the courses found",{c['name']})
    search_term = input("🔎 Enter part of your course name: ")
    selected = select_course_by_name(courses, search_term)
    if selected:
        print(f"Selected course: {selected['name']}")
    else:
      print("Incorrect course name")
