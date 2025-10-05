import re
from bs4 import BeautifulSoup
import aiohttp
import random
import time
import asyncio
import os
import json
from datetime import datetime

from typing import List, Dict
from llama_index.core.node_parser import SentenceSplitter
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI

api_key = os.getenv("api_key")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0"
]

# ----------------- Async get_soup -----------------
async def async_get_soup(session, url, max_retries=5, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive"
            }
            async with session.get(url, headers=headers, timeout=30) as resp:
                if resp.status == 467:
                    wait = backoff_factor ** attempt + random.uniform(1, 5)
                    print(f"⚠️ Blocked (467). Retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                html = await resp.text()
                return BeautifulSoup(html, "html.parser")

        except Exception as e:
            wait = backoff_factor ** attempt + random.uniform(1, 3)
            print(f"⚠️ Request error {e}. Retrying in {wait:.1f}s...")
            await asyncio.sleep(wait)

    print(f"❌ Failed to fetch {url} after {max_retries} attempts.")
    return None


def save_to_json(new_data, name, unique_keys=("act_title", "section"), mode="unique"):
    """
    Save data to JSON file with optional uniqueness check.

    Args:
        new_data (dict | list): Data to insert (single dict or list of dicts).
        name (str): File name (without .json extension).
        unique_keys (tuple): Keys used to determine uniqueness (only in 'unique' mode).
        mode (str): 'unique' -> replace based on unique_keys,
                    'append' -> simply append all data.
    """
    file_path = f"{name}.json"

    # Step 1: Load existing data if file exists
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as infile:
            try:
                all_data = json.load(infile)
                if not isinstance(all_data, list):  # defensive
                    all_data = []
            except json.JSONDecodeError:
                all_data = []
    else:
        all_data = []

    # Step 2: Normalize new_data into list
    if isinstance(new_data, dict):
        new_data = [new_data]

    if mode == "unique":
        # Build a lookup dictionary for fast replacement
        data_dict = {
            tuple(item.get(k) for k in unique_keys): item
            for item in all_data
        }

        # Insert or replace
        for entry in new_data:
            key = tuple(entry.get(k) for k in unique_keys)
            data_dict[key] = entry

        final_data = list(data_dict.values())

    elif mode == "append":
        # Just extend without uniqueness checking
        final_data = all_data + new_data

    else:
        raise ValueError("Invalid mode. Use 'unique' or 'append'.")

    # Step 3: Save back to JSON
    with open(file_path, "w", encoding="utf-8") as outfile:
        json.dump(final_data, outfile, indent=4, ensure_ascii=False)


def extract_section(text: str) -> str | None:
    """
    Extract section identifiers like:
    (1), (12), (14A), (15B), (1AA), (12AB), etc.
    with optional leading em-dash (—).
    """
    match = re.match(r'^\s*—?\(\d+[A-Z]*\)', text.strip())
    if match:
        return match.group(0).replace("—", "")
    return ""

def build_prompt(context) :
    question = f"""
You are an expert tax lawyer and legal knowledge engineer. Your task is to transform the given legal context (provided as a JSON object) into a structured JSON output. 

### Instructions:
1. From the field "text":
   - Create "section_summary": a concise, semantic summary of the legal meaning of this section. Maximum 3 sentences. Preserve obligations, conditions, and exceptions.
   - Create "questions_this_excerpt_can_answer": a **single string** containing exactly three bullet points. Each bullet point must be a potential question a lawyer could ask that this section helps answer. Each question must be precise and self-contained.
   - If the text states "Repealed" or "Revoked", reflect this clearly in the summary and generate repeal-related questions (e.g., "When was this section repealed?").
   - Do not invent facts outside the provided text.

2. Output must be **only valid JSON**. No extra commentary, no explanations, no code fences.

---

### Expected Output Format:

{{
  "section_summary": "...",
  "questions_this_excerpt_can_answer": "1. Question 1\n2. Question 2\n3. Question 3"
}}

---

for context: {context}


"""
    return question

class LegalOutput(BaseModel):
    section_summary: str
    questions_this_excerpt_can_answer: str

# Structured LLM
llm = OpenAI(model="gpt-4o-mini", api_key= api_key)
sllm = llm.as_structured_llm(output_cls=LegalOutput)


def clean_text(text: str) -> str:
    # Step 1: Replace non-breaking spaces with normal space
    text = text.replace("\xa0", " ")

    # Step 3: Strip leading/trailing spaces
    return text.strip()

def built_dict (act_title, section, url, final, act_type, full_sub) : 
    return {
            "act_name" : act_title,
            "section" :  f"{section}",
            "url" : url, 
            "text" : final, 
            "is_subsection_of" : f"{act_title} {section}",
            "act_type" : act_type,
            "section_title" : full_sub
            }


# --- async enrichment ---
async def get_enrich_async(context_dict, semaphore):
    """Enrich context_dict with LLM output using async + rate limiting."""
    async with semaphore:  # limit concurrent API calls
        prompt = build_prompt(context_dict)
        loop = asyncio.get_event_loop()
        
        # run blocking call in executor (non-blocking for others)
        resp = await loop.run_in_executor(None, lambda: sllm.complete(prompt))
        
        resp_dict = json.loads(str(resp))
        fix = {**context_dict, **resp_dict}
        print("✅ enrichment done")
        return 
splitter = SentenceSplitter(
        chunk_size=2096,
        chunk_overlap=210,
        paragraph_separator="\n\n"
    )

async def get_chunking_async(act, sec, href, long_text, type_act, data_part, semaphore):
    """Split into chunks (if needed) and enrich all chunks asynchronously."""
    results = []

    if len(long_text) > 10000:
        sub_chunks = splitter.split_text(long_text)

        tasks = []
        for idx, sub in enumerate(sub_chunks, start=1):
            chunk_section = f"{sec}-[{idx}]"
            chunk_text = sub.strip()
            chunk = built_dict(act, chunk_section, href, chunk_text, type_act, data_part)
    
            # schedule enrichment task
            #tasks.append(get_enrich_async(chunk, semaphore))
            tasks.append(chunk)

        # run all enrichments concurrently
        enriched_chunks = await asyncio.gather(*tasks)
        print(f"✅ {len(enriched_chunks)} chunks enriched")
        results.extend(enriched_chunks)

    else:
        result = built_dict(act, sec, href, long_text, type_act, data_part)
        #enriched = await get_enrich_async(result, semaphore)
        enriched = result
        results.append(enriched)
        print("✅ single enrichment done")

    return results



# ----------------- async get_maps -----------------
async def async_get_maps(session, soup, url):
    base_ids = soup.find("nav", class_="navbar navbar-light bg-white flex-column align-items-stretch toc")
    ids = base_ids.find_all("a")
    check_urls = [a['href'] for a in ids if a.has_attr("href")]

    pattern = re.compile(r"^#P\d+-$")

    if any(pattern.match(u) for u in check_urls):
        codes = [u for u in check_urls if pattern.match(u)]
        return {
            "codes": codes,
            "type": "main act",
            "url": url
        }
    else:
        legislation = base_ids.find("nav", class_="nav nav-pills flex-column")
        codes = []
        try:
            ids_legislation = legislation.find_all("a")
            for a in ids_legislation:
                if a.has_attr("href"):
                    codes.append(a['href'])
        except Exception:
            codes.append("#av-")
        return {
            "codes": codes,
            "type": "subsidiary",
            "url": url
        }


# ----------------- async scrape -----------------
async def async_scrape(session, types: dict, act_type, json_name, semaphore, link=""):
    all_data = []

    # ================= Subsidiary =================
    if types["type"] == "subsidiary":
        url = f"{link}{types['codes'][0]}"
        print(url)
        soup = await async_get_soup(session, url)
        if not soup:
            return []

        act_title = soup.find(
            "nav",
            class_="navbar navbar-light bg-white flex-column align-items-stretch toc"
        ).find("a").find("b").get_text(separator=" ", strip=True)
        act_title = " ".join(act_title.split())
        print(act_title)

        part1 = soup.find("td", class_="SLActHd")
        part2 = soup.find("div", class_="empProvHd")
        data_part_title = f"{part1.get_text(strip=True) if part1 else ''} {part2.get_text(strip=True) if part2 else ''}".strip()

        base = soup.find("div", class_="body")

        # prov1Rep
        tables_rep = base.find_all("div", class_="prov1Rep")
        if tables_rep:
            for table in tables_rep:
                deept = table.find("td").find("em") or table.find("td").find("i")
                section = table.find("strong").text.strip().replace(".", "")
                final = deept.text.strip() if deept else "deleted"
                fix = await get_chunking_async(act_title, section, url, final, act_type, data_part_title, semaphore)
                print(fix)
                all_data.extend(fix)
                save_to_json(fix, json_name, mode="append")

        # prov1 (normal structure)
        tables = base.find_all("div", class_="prov1")
        if not tables:
            unwanted_span = soup.find("div", class_="amendNote")
            if unwanted_span:
                unwanted_span.decompose()
            content = soup.find("div", class_="front").find("td", class_="openWd").text.strip()
            section = "1"
            fix = await get_chunking_async(act_title, section, url, content, act_type, data_part_title, semaphore)
            print(fix)
            all_data.extend(fix)
            save_to_json(fix, json_name, mode="append")
        else:
            for table in tables:
                deept = table.find("td", class_="prov1Txt") or table.find("td", class_="prov1RepTxt")
                section = table.find("strong").text.strip().replace(".", "")
                #header = table.find("td", class_="prov1Hdr").text.strip() # INCASE MAU PAKE SUB HEADER E.G"CITATION" , "DEFINITION"
                #print(header)
                try : 
                    content = table.find_all("table")[1].text.strip()
                except : 
                    content = table.find("table").text.strip() # 1 ayat full
                full_content = clean_text(content)
                #fix = built_dict(act_title, section, url, data_part, full_content, act_type, data_part_title)
                fix = await get_chunking_async(act_title, section, url, full_content, act_type, data_part_title, semaphore)
                print(fix)
                all_data.extend(fix)

                deeptable = deept.find("span", class_="prov2TxtIL")
                sub_deept = deept.find_all("td", class_="prov2Txt")

                if deeptable and sub_deept:
                    unwanted_span = deeptable.find("div", class_="amendNote")
                    if unwanted_span:
                        unwanted_span.decompose()

                    finala = deeptable.text.strip()
                    final = clean_text(finala)
                    sub_section_awal = extract_section(final)
                    section_awal = f"{section}{sub_section_awal}" 
                    print(f"{section_awal} ")
                    #fix = built_dict(act_title, section_awal, url, data_part, final, act_type, data_part_title)
                    fix = await get_chunking_async(act_title, section_awal, url, final, act_type, data_part_title, semaphore)
                    print(fix)
                    all_data.extend(fix)
                    save_to_json(fix, json_name, mode="append")
                    print(final)
                    print(f"{section_awal}")
                    print("x" * 50)

                    for deep in sub_deept:
                        unwanted_span = deep.find("div", class_="amendNote")
                        if unwanted_span:
                            unwanted_span.decompose()
                        finala = deep.text.strip()
                        final = clean_text(finala)
                        sub_section = extract_section(final)
                        section_final = f"{section}{sub_section}"

                        #fix = built_dict(act_title, section_final, url, data_part, final, act_type, data_part_title)
                        fix = await get_chunking_async(act_title, section_final, url, final, act_type, data_part_title, semaphore)
                        print(fix)
                        all_data.extend(fix)
                        save_to_json(fix, json_name, mode="append")
                     
                        print(f"{section_final} if deeptable and sub_deept")
                        print("=" * 50)

                elif not deeptable:
                    unwanted_span = deept.find("div", class_="amendNote")
                    if unwanted_span:
                        unwanted_span.decompose()
                    finala = deept.text.strip()
                    final = clean_text(finala)
                    sub_section = extract_section(final)
                    section_final = f"{section}{sub_section}"

                    #fix = built_dict(act_title, section_final, url, data_part, final, act_type, data_part_title)
                    fix = await get_chunking_async(act_title, section_final, url, final, act_type, data_part_title, semaphore)
                    print(fix)
                    all_data.extend(fix)
                    save_to_json(fix, json_name, mode="append")
                    print(final)
                    print(f"{section_final} elif deeptable")
                    print("#" * 50)
                else:
                    continue
                
        return all_data

    # ================= Main Act =================
    else:
        all_data = [] 
        all_codes = types["codes"]
        base_url = types["url"]

        # fetch all detail soups concurrently
        tasks = []
        urls = []
        for code in all_codes:
            prefix_url = code.replace("#", "").replace("-", "")
            url_final = f"{base_url}?ProvIds&ProvIds={prefix_url}-{code}"
            urls.append(url_final)
            tasks.append(async_get_soup(session, url_final))

        soups = await asyncio.gather(*tasks)

        for url_final, soup in zip(urls, soups):
            if not soup:
                continue

            act_title = soup.find(
                "nav", class_="navbar navbar-light bg-white flex-column align-items-stretch toc"
            ).find("a").find("b").get_text(separator=" ", strip=True)
            act_title = " ".join(act_title.split())

            hdr = soup.find("div", class_="partNo").text.strip()
            no_element = soup.find("td", class_="partHdr")
            no = no_element.text.strip() if no_element else soup.find("td", class_="part").find("table").text.strip()
            data_part_title = f"{hdr} {no}"

            base = soup.find("td", class_="part")
            tables_rep = base.find_all("div", class_="prov1Rep")
            for table in tables_rep:
                deept = table.find("td").find("i")
                section = table.find("strong").text.strip().replace(".", "")
                finala = deept.text.strip()
                final = clean_text(finala)
                sub_section = extract_section(final)
                section_final = f"{section}{sub_section}"

                fix = await get_chunking_async(act_title, section_final, url_final, final, act_type, data_part_title, semaphore)
                all_data.extend(fix)
                
            tables = base.find_all("div", class_= "prov1")
            for table in tables : 
                deept = table.find("td", class_="prov1Txt") 
                
                section = table.find("strong").text.strip().replace(".","")
                #print(len(deept))
                
                quote_table = deept.find_all("td", class_ = "QEProv1Txt")
                if quote_table : 
                    print("ada nih") 
                    unwanted_span = deeptable.find('div', class_='amendNote')
                    if unwanted_span:
                        unwanted_span.decompose()
                    finala = deept.text.strip()
                    final = clean_text(finala)
                    sub_section = extract_section(final)
                    section_final = f"{section}{sub_section}"
                    #print(final)
                    print(f"section : {section}{sub_section}")
                    print("%"*50)
                    fix = await get_chunking_async(act_title, section_final, url_final, final, act_type, data_part_title, semaphore)
                    all_data.extend(fix)
                    continue
                    
                deeptable = deept.find("span", class_="prov2TxtIL")
                if deeptable:
                    unwanted_span = deeptable.find('div', class_='amendNote')
                    if unwanted_span:
                        unwanted_span.decompose()
                    finala = deeptable.text.strip()
                    final = clean_text(finala)
                    """
                    if filter_data(final) == True: 
                        continue
                    """
                    sub_section = extract_section(final)
                    section_final = f"{section}{sub_section}"
                    #print(final)
                    print(f"section : {section}{sub_section}")
                    print("#"*50)
                    fix = await get_chunking_async(act_title, section_final, url_final, final, act_type, data_part_title, semaphore)
                    all_data.extend(fix)
       
                deeptables = deept.find_all("td", class_="prov2Txt")
                if deeptables : 
                    for deep in deeptables :
                        unwanted_span = deep.find('div', class_='amendNote')
                        if unwanted_span:
                            unwanted_span.decompose()
                        finala = deep.text.strip()
                        final = clean_text(finala)
                        """
                        if filter_data(final) == True: 
                            continue
                        """
                        sub_section = extract_section(final)
                        section_final = f"{section}{sub_section}"
                        #print(final)
                        print(f"section : {section}{sub_section}")
                        print("*"*50)
                        
                        fix = await get_chunking_async(act_title, section_final, url_final, final, act_type, data_part_title, semaphore)
                        all_data.extend(fix)
                else : 
                    finala = deept.text.strip()
                    final = clean_text(finala)
                    """
                    if filter_data(final) == True: 
                        continue
                    """
                    sub_section = extract_section(final)
                    section_final = f"{section}{sub_section}"
                    #print(final)
                    print(f"section : {section}{sub_section}")
                    print("="*50)
                    fix = await get_chunking_async(act_title, section_final, url_final, final, act_type, data_part_title, semaphore)
                    all_data.extend(fix)
                print("-"*40)
            # (⚡ keep the rest of your prov1 parsing here exactly — omitted for brevity)

        save_to_json(all_data, json_name, mode="append")
        return all_data

    
# ----------------- Per-URL processing -----------------
async def process_url(session, link, semaphore):
    print(f"processing: {link}")
    soup1 = await async_get_soup(session, link)
    if not soup1:
        return []

    codes = await async_get_maps(session, soup1, link)
    print(codes)

    all_data = await async_scrape(session, codes, "Subsidiary Legislation", "subsidiary_new", semaphore, link) if codes["type"] == "subsidiary" else await async_scrape(session, codes, "Main Act", "main_act_new", "", semaphore)
    return all_data


# ----------------- Async main -----------------
async def main():
    with open('dataset/subsidiary_links.json', 'r', encoding="utf-8") as f:
        urls = json.load(f)

    start_time = time.time()
    start_at = datetime.now().strftime("%H:%M:%S")
    print(f"start at : {start_at}")

    semaphore = asyncio.Semaphore(5) 
    final_data = []
    async with aiohttp.ClientSession() as session:
        tasks = [process_url(session, link, semaphore) for link in urls[:10]]
        results = await asyncio.gather(*tasks)

    for res in results:
        final_data.extend(res)

    print(f"total extracted data : {len(final_data)}")
    end_at = datetime.now().strftime("%H:%M:%S")
    print(f"end at : {end_at}")
    print(f"elapsed : {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    asyncio.run(main())