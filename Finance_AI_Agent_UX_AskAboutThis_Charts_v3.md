# Finance AI-Agent UX: Interaktiv tabell → Fråga → Undersök → Svara

## Syfte
Skapa en upplevelse där tabeller och grafer inte är “slutrapport”, utan en interaktiv arbetsyta där användaren kan peka på en siffra och omedelbart be agenten undersöka den. Resultatet ska kännas som en mänsklig analytiker: snabb, kontextmedveten, konsekvent, och spårbar.

## Önskad känsla (north star)
- Användaren ska känna: **“Jag jobbar i en analysyta, inte i en chat.”**
- Varje siffra ska kännas **klickbar och undersökningsbar**.
- Varje svar ska kännas **förankrat i data**, med tydlig källa och möjlighet att “drilla vidare”.
- Interaktioner ska vara **reversibla** och **reproducerbara** (revision/lineage i bakgrunden, men inte i vägen).

---

## 1) “Ask about this” (Markera cell/rad → skicka undersökningsuppdrag)

### 1.1 Hur användaren upplever det
Användaren tittar på en tabell eller graf och ser en avvikelse, en topp, en ovanlig trend eller en siffra som inte känns rimlig. I stället för att behöva formulera en lång fråga kan användaren:

- Klicka/markera en **cell** (en specifik siffra)
- Markera en **rad** (en post/kategori över flera perioder)
- Markera ett **intervall** (t.ex. flera periodkolumner eller flera rader)

Därefter väljer användaren i en kontextmeny: **“Ask about this”**.

Detta öppnar en liten “frågepanel” som:
- Autogenererar en bra fråga baserat på valet (användaren kan redigera)
- Visar exakt *vad* som valts (för att undvika missförstånd)
- Föreslår vanliga uppföljningar (“drivers”, “drilldown”, “jämför”, “graf”)

### 1.2 Vad användaren kan be om (exempel på “job-to-be-done”)
**För en markerad cell** (t.ex. kostnad konto X i 2025-11):
- “Förklara den här siffran”
- “Varför är den högre än förra månaden?”
- “Är detta en engångseffekt?”
- “Vad är de största underliggande raderna/transaktionerna?”

**För en markerad rad** (t.ex. konto/kategori över tid):
- “Sammanfatta trenden och brytpunkter”
- “Vilka månader sticker ut och varför?”
- “Jämför årets YTD mot föregående år”

**För variance-tabell:**
- “Vilka drivare förklarar avvikelsen?”
- “Bryt ner på leverantör/enhet/projekt”
- “Visa vilka rader som står för 80% av avvikelsen”

**För en graf** (t.ex. trendlinje):
- “Varför faller den här kurvan efter oktober?”
- “Vilka kategorier driver uppgången?”

### 1.3 Hur agenten ska bete sig (svart låda ur användarens perspektiv)
Efter att användaren tryckt “Ask about this” ska agenten:

1) Bekräfta kontexten kort: “Jag undersöker [det här valet].”
2) Utföra relevanta undersökningar automatiskt:
   - Drilldown till lämplig granularitet
   - Jämförelser mot referensperiod
   - Identifiera topp-drivers och avvikelser
3) Returnera ett svar i tre lager:
   - **TL;DR** (1–2 meningar)
   - **Drivers** (bullets med siffror, top 3–8)
   - **Evidence** (tabell/graf med underlag)
4) Erbjuda naturliga nästa steg:
   - “Visa fler transaktioner”
   - “Filtrera bort engångsposter”
   - “Bryt ned per leverantör/enhet”
   - “Skapa en waterfall/graf”

**Kärnprincip:** användaren ska kunna gå från “jag ser en siffra” till “jag förstår varför” på 1–2 interaktioner.

### 1.4 Viktiga UX-principer för “Ask about this”
- **Low friction:** Användaren ska inte behöva skriva om dimensioner/perioder.
- **No ambiguity:** UI visar alltid exakt vad selection avser.
- **Actionable output:** Svar ska leda till nästa klickbara artefakt, inte bara text.
- **Progressive disclosure:** Standard är kort svar + möjlighet att expandera.
- **Data-first:** Svar ska alltid ha kvantifierad evidens (tabell/graf), inte bara narrativ.

---

## Charts (Plotly) – “4 tillåtna grafer” som känns CFO-grade

### Målbild
Grafer ska kännas som naturliga “views” av samma analys, inte som extra exportsteg. Användaren ska kunna skapa en graf från tabell med ett klick, och vice versa.

### 4 standardgrafer (whitelist)
1) **Trend (Line)**
   - För tidserier: “Hur utvecklas detta över tid?”
   - Stöd för att färga per kategori (om det finns rimliga dimensioner)

2) **Top drivers (Bar)**
   - “Vilka står för mest (senaste period / vald period)?”
   - Stöd för top_n och “Others”-bucket

3) **Composition (Stacked)**
   - “Hur är totalen sammansatt över tid?”
   - För kostnadsslag/kategorier

4) **Variance Waterfall (Waterfall)**
   - “Hur går vi från A till B – vilka drivare förklarar skillnaden?”
   - Särskilt för variance tool, men kan också användas generellt när två perioder jämförs

### Hur användaren skapar grafer
- En tydlig knapp: **“Create chart”** eller “Visualize”.
- Alternativt via kontextmeny: “Skapa graf av detta”.
- UI föreslår vilken av de fyra som passar bäst, men användaren kan byta.

### Hur användaren interagerar med grafer
- Hover ger tydlig tooltip med dimensioner + värde.
- Klick på datapunkt/serie ska kunna trigga **Ask about this** (samma koncept som tabell).
- En graf ska kunna “driva” filter (t.ex. klicka på en kategori för att filtrera tabellen).

---

## “Ask about this” + Charts: helhetsupplevelsen
Det viktigaste är att tabell/graf och agenten känns sammanvävda:

- Tabell → “Ask about this” → Agent svarar med en graf som underlag.
- Graf → “Ask about this” → Agent svarar med en tabell med drilldown.
- Svar innehåller alltid tydliga nästa steg som är klickbara.

I praktiken: varje undersökning ska kunna bli en ny “vy” av datan.

---

## Designprinciper att hålla fast vid
- **One-click investigation:** all analys börjar med markering.
- **Guardrails och förutsägbarhet:** agenten gör smarta val men förklarar vad den gjorde.
- **Reproducerbarhet:** en undersökning ska kunna återbesökas och upprepas.
- **Template-driven charts:** hellre få grafer som alltid är bra än “allt är möjligt” men inkonsekvent.
- **Scale gracefully:** fungerar både för särskilda finance-tools och generella SQL-resultat, men med olika nivå av autopilot.

---

## Definition of done (när funktionen “känns klar”)
- Användaren kan markera cell/rad/grafpunkt och trigga “Ask about this”.
- Agenten returnerar ett svar som alltid innehåller:
  - kort sammanfattning
  - kvantifierade drivers
  - evidens i tabell/graf
- Användaren kan fortsätta drilla 2–3 nivåer utan att tappa bort kontext.
- Upplevelsen känns som en sammanhängande analysyta, inte chat-fragment.

---

## Staging plan: Hur mycket göra i Streamlit innan Lovable?

### Målsättning
Använd Streamlit för att **sätta kontrakt, pipelines och end-to-end-flöden**. Undvik att bygga “häftig” interaktion där, eftersom den ändå ska göras om i Lovable.

### Gör i Streamlit innan Lovable (rekommenderat)
1) **“Ask about this” end-to-end (utan perfekt UX)**
   - En enkel markering av rad/cell (kan vara via klick + state eller dropdown för rad-id/cell).
   - En knapp **Ask about this** som skickar selection-kontekst till agenten.
   - Verifiera att agenten kan undersöka och returnera nya artifacts (tabell/graf) kopplade till samma session/turn.

2) **Inför `presentation_chart` som artifact-typ (Plotly)**
   - Skapa chart-artefakter på samma sätt som tabeller: “chart som vy av redan loggad data”.
   - Rendera i Streamlit för att bevisa att artifact-flödet fungerar stabilt även för grafer.

3) **Implementera 4 chart-templates (whitelist)**
   - Trend (line)
   - Top drivers (bar)
   - Composition (stacked)
   - Variance waterfall (waterfall)
   - I Streamlit räcker en dropdown “Chart template” + “Create chart”.

4) **Basstöd för generellt SQL-verktyg**
   - Testa att grafer fungerar även för generiska tabeller, inte bara för income statement/variance.
   - Om auto-detektion inte räcker: låt användaren välja x/y (och ev category) i Streamlit.

5) **Minimal explainability**
   - Visa tydligt under tabell/graf: vilka transformationssteg som applicerats, och eventuella notes/guardrails.
   - Målet är att kontrollera att backend producerar stabil metadata för senare “Explained UI”.

### Skjut till Lovable (bör inte göras i Streamlit)
- Kontextmeny/högerklick på celler, multi-select, drag-and-drop och “command bar”.
- Cross-filter mellan graf ↔ tabell.
- Polerad “Ask about this”-panel med förslag, inline-edit och snabba uppföljningar.
- All “premium”-känsla i direktmanipulationen.

### Stoppkriterium: När är det dags att gå till Lovable?
Gå vidare när detta är uppfyllt:
- **Ask about this** fungerar end-to-end (selection → agent → nya artifacts).
- **Plotly charts** kan sparas och renderas som artifacts.
- Alla **4 chart-templates** fungerar för:
  - income statement tool
  - variance tool
  - minst ett generellt SQL-resultat (med enkla val om nödvändigt)
- UX i Streamlit är “tillräcklig för test”, men ingen tid läggs på finpolering.

---

## V1/V2 strategi: Validera kärnnytta innan du skalar upp

### Bakgrund och mål
Det finns tydligt värde i att göra en **V1 utan “Ask about this” och utan grafer**, för att först validera att finance-agenten fungerar i praktiken (korrekt data, stabila tools, begripligt beteende) innan du bygger “wow”-funktioner.

Kärnprincip: **skala inte upp UX/orkestrering förrän du vet att agentens kärnnytta är verklig**.

### Rekommenderad strategi
Bygg en V1 som bevisar kärnnyttan, och håll V2 som strikt isolerade moduler/spikes bakom feature flags.

#### V1 (utan Ask/Charts): “Product-ready tables”
V1 ska kännas som en produktklient (Lovable), men vara enkel och robust:

- **Chat + artifacts feed**: tydlig historik per turn, klicka in på en turn och se tabell-artefakten.
- **Formatering/redo**: unit, decimals, sort, top_n, filter (det du redan har).
- **Undo/back i formatering**: gå tillbaka/fram i lineage.
- **Minimal explainability**: visa transformationssteg + notes/guardrails tydligt under tabellen.
- **Export & delning**: copy TSV/CSV, download CSV, copy “commentary” (om du har narrativ).
- **Stabil felhantering**: tydligt när data saknas eller filter ger 0 rader, och hur man tar sig vidare.

Målet är att V1 ska vara **tillräckligt bra för vardagsanvändning** så att du kan utvärdera nyttan på riktigt.

#### V2 (Ask/Charts): “Investigation workflows”
V2 ska byggas utan att störa V1:

- **Ask about this** (selection → investigation → nya artifacts)
- **Plotly charts** som artifacts (4 templates: Trend, Top drivers, Composition, Waterfall)
- Cross-filter och premium direktmanipulation först när V1 är bevisad

Tekniskt/organisatoriskt:
- Lägg V2 bakom feature flags.
- Bygg V2 som fristående moduler: `selection payload` + `presentation_chart` + investigation workflow.
- Timeboxa V2-spikes så de inte konkurrerar med V1:s stabilitet.

### Är det värt att ta V1 till Lovable ändå?
Ja, **om** du gör Lovable till en “thin client” över din backend och undviker att fastna i frontend-detaljer.

Lovable i V1 hjälper dig att:
- få en produktkänsla som gör feedback mer realistisk,
- standardisera API/kontrakt mellan backend och UI,
- fortsätta använda Streamlit parallellt som debug-konsol.

### Testa att agenten “funkar” innan du skalar upp
Sätt en enkel V1-checklista med 10–20 standardfrågor som motsvarar dina vanligaste use cases. Exempel:

- “Visa resultaträkning YTD för 2025-11”
- “Vilka är top drivers i kostnadsökningen mot föregående år?”
- “Varför är konto X högt i november?”
- “Bryt ner avvikelse på enhet/leverantör”

**Pass/Fail-kriterier (exempel):**
- Tabell är korrekt och konsekvent mellan körningar.
- Svar refererar till faktiska siffror i tabellen.
- Du kan lösa en vardagsfråga snabbare än i Excel/BI.
- Systemet beter sig begripligt när data saknas eller filter ger 0 rader.

När V1 klarar detta: bygg V2-funktionerna för att öka “wow” och effektivitet.

