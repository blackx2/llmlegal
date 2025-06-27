import asyncio
import json
import re
from playwright.async_api import async_playwright

combined_keywords = "reajuste cassi"
highlight_in_url = "reajuste,cassi"

def segmentar_decisao(texto):
    lines = texto.splitlines()
    jurisprudencia = []
    relatorio = []
    voto = []
    outros = []

    current_section = "outros"
    
    for line in lines:
        clean_line = line.strip()

        # Check for jurisprudência-like indentation (5+ leading spaces)
        if re.match(r"^\s{5,}", line):
            jurisprudencia.append(clean_line)
            continue

        # Check for section starts
        if re.search(r"^\s*(RELATÓRIO|Relatório|Relata-se)", line):
            current_section = "relatorio"
        elif re.search(r"^\s*(VOTO|Voto|É o voto|Voto do relator)", line):
            current_section = "voto"

        # Assign content to the correct section
        if current_section == "relatorio":
            relatorio.append(clean_line)
        elif current_section == "voto":
            voto.append(clean_line)
        elif current_section == "outros":
            outros.append(clean_line)

    return {
        "jurisprudencia": "\n".join(jurisprudencia).strip(),
        "relatorio": "\n".join(relatorio).strip(),
        "voto": "\n".join(voto).strip(),
        "outros": "\n".join(outros).strip()
    }

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://pje-jurisprudencia.tjpb.jus.br/")
        await page.fill("#inputEmenta", combined_keywords)
        await page.click("button.btn-primary[type='submit']")
        await page.wait_for_selector("table.table tbody tr")

        page_number = 1
        output_file = open("decisions.jsonl", "a", encoding="utf-8")

        while True:
            print(f"\n🔎 Página {page_number}")
            rows = await page.query_selector_all("table.table tbody tr")

            for idx, row in enumerate(rows, start=1):
                ementa = await row.query_selector("blockquote p.highlight-words")
                numero = await row.query_selector("blockquote small")
                base_link = await row.query_selector("a.btn-default")

                ementa_text = await ementa.inner_text() if ementa else ""
                numero_text = await numero.inner_text() if numero else ""
                base_href = await base_link.get_attribute("href") if base_link else ""
                full_link = base_href.split("?")[0] + f"?words={highlight_in_url}" if base_href else ""

                # Open detail page
                detail_page = await context.new_page()
                await detail_page.goto(full_link)
                try:
                    await detail_page.wait_for_selector(".inteiro-teor .panel-body", timeout=30000)
                    await asyncio.sleep(3)
                    teor_div = await detail_page.query_selector(".inteiro-teor .panel-body")
                    inteiro_teor_text = await teor_div.inner_text() if teor_div else ""

                    if inteiro_teor_text.strip():
                        partes = segmentar_decisao(inteiro_teor_text)
                        record = {
                            "numero": numero_text,
                            "ementa": ementa_text,
                            "link": full_link,
                            "conteudo": {
                                "jurisprudencia": partes["jurisprudencia"],
                                "relatorio": partes["relatorio"],
                                "voto": partes["voto"],
                                "outros": partes["outros"]
                            }
                        }
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"⚠️ Erro ao carregar inteiro teor: {e}")
                await detail_page.close()
                await asyncio.sleep(1)

            # Próxima página
            next_button = await page.query_selector("#prox")
            parent_li = await next_button.evaluate_handle("el => el.parentElement")
            class_name = await parent_li.get_property("className")

            if "disabled" in await class_name.json_value():
                print("📄 Última página.")
                break
            else:
                await next_button.click()
                page_number += 1
                await asyncio.sleep(2)
                await page.wait_for_selector("table.table tbody tr")

        output_file.close()
        await browser.close()

asyncio.run(run())
