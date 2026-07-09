"""S&P 500 ticker listesi ve sektör bilgisi."""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

CACHE = DATA_DIR / "sp500_tickers.parquet"


def fetch_sp500_tickers(use_cache: bool = True) -> pd.DataFrame:
    """
    Wikipedia'dan S&P 500 bileşenlerini çeker.
    Kolonlar: Symbol, Security, GICS Sector, GICS Sub-Industry
    """
    if use_cache and CACHE.exists():
        return pd.read_parquet(CACHE)

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        # Kolon isimlerini normalize et
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={
            "Symbol": "Symbol",
            "Security": "Name",
            "GICS Sector": "Sector",
            "GICS Sub-Industry": "SubIndustry",
        })
        # BRK.B → BRK-B (yfinance formatı)
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        df = df[["Symbol", "Name", "Sector", "SubIndustry"]].dropna(subset=["Symbol"])
        df.to_parquet(CACHE, index=False)
        return df
    except Exception as e:
        print(f"  Wikipedia'dan cekilemedi: {e}")
        # Fallback: en büyük 100 hisse sabit liste
        return _fallback_list()


def _fallback_list() -> pd.DataFrame:
    """200+ büyük ABD hissesi — sektör dengeli."""
    rows = [
        # ── Information Technology ──────────────────────────────────────
        ("AAPL","Apple","Information Technology"),
        ("MSFT","Microsoft","Information Technology"),
        ("NVDA","NVIDIA","Information Technology"),
        ("AVGO","Broadcom","Information Technology"),
        ("AMD","AMD","Information Technology"),
        ("INTC","Intel","Information Technology"),
        ("CRM","Salesforce","Information Technology"),
        ("ORCL","Oracle","Information Technology"),
        ("ADBE","Adobe","Information Technology"),
        ("NOW","ServiceNow","Information Technology"),
        ("CSCO","Cisco","Information Technology"),
        ("ACN","Accenture","Information Technology"),
        ("IBM","IBM","Information Technology"),
        ("QCOM","Qualcomm","Information Technology"),
        ("TXN","Texas Instruments","Information Technology"),
        ("MU","Micron Technology","Information Technology"),
        ("AMAT","Applied Materials","Information Technology"),
        ("LRCX","Lam Research","Information Technology"),
        ("KLAC","KLA Corp","Information Technology"),
        ("SNPS","Synopsys","Information Technology"),
        ("CDNS","Cadence Design","Information Technology"),
        ("HPQ","HP Inc","Information Technology"),
        ("DELL","Dell Technologies","Information Technology"),
        ("PANW","Palo Alto Networks","Information Technology"),
        ("FTNT","Fortinet","Information Technology"),
        # ── Communication Services ──────────────────────────────────────
        ("META","Meta","Communication Services"),
        ("GOOGL","Alphabet A","Communication Services"),
        ("GOOG","Alphabet C","Communication Services"),
        ("NFLX","Netflix","Communication Services"),
        ("DIS","Disney","Communication Services"),
        ("T","AT&T","Communication Services"),
        ("VZ","Verizon","Communication Services"),
        ("TMUS","T-Mobile","Communication Services"),
        ("CMCSA","Comcast","Communication Services"),
        ("CHTR","Charter Comm","Communication Services"),
        ("EA","Electronic Arts","Communication Services"),
        ("TTWO","Take-Two","Communication Services"),
        # ── Consumer Discretionary ──────────────────────────────────────
        ("AMZN","Amazon","Consumer Discretionary"),
        ("TSLA","Tesla","Consumer Discretionary"),
        ("HD","Home Depot","Consumer Discretionary"),
        ("MCD","McDonald's","Consumer Discretionary"),
        ("NKE","Nike","Consumer Discretionary"),
        ("SBUX","Starbucks","Consumer Discretionary"),
        ("LOW","Lowe's","Consumer Discretionary"),
        ("TGT","Target","Consumer Discretionary"),
        ("BKNG","Booking Holdings","Consumer Discretionary"),
        ("ABNB","Airbnb","Consumer Discretionary"),
        ("UBER","Uber","Consumer Discretionary"),
        ("GM","General Motors","Consumer Discretionary"),
        ("F","Ford","Consumer Discretionary"),
        ("ROST","Ross Stores","Consumer Discretionary"),
        ("TJX","TJX Companies","Consumer Discretionary"),
        # ── Consumer Staples ────────────────────────────────────────────
        ("WMT","Walmart","Consumer Staples"),
        ("PG","Procter & Gamble","Consumer Staples"),
        ("KO","Coca-Cola","Consumer Staples"),
        ("PEP","PepsiCo","Consumer Staples"),
        ("COST","Costco","Consumer Staples"),
        ("PM","Philip Morris","Consumer Staples"),
        ("MO","Altria","Consumer Staples"),
        ("MDLZ","Mondelez","Consumer Staples"),
        ("CL","Colgate-Palmolive","Consumer Staples"),
        ("KHC","Kraft Heinz","Consumer Staples"),
        ("GIS","General Mills","Consumer Staples"),
        ("HSY","Hershey","Consumer Staples"),
        ("KR","Kroger","Consumer Staples"),
        ("SYY","Sysco","Consumer Staples"),
        # ── Health Care ─────────────────────────────────────────────────
        ("UNH","UnitedHealth","Health Care"),
        ("LLY","Eli Lilly","Health Care"),
        ("JNJ","Johnson & Johnson","Health Care"),
        ("ABBV","AbbVie","Health Care"),
        ("MRK","Merck","Health Care"),
        ("ABT","Abbott Labs","Health Care"),
        ("TMO","Thermo Fisher","Health Care"),
        ("DHR","Danaher","Health Care"),
        ("BMY","Bristol-Myers","Health Care"),
        ("AMGN","Amgen","Health Care"),
        ("GILD","Gilead","Health Care"),
        ("ISRG","Intuitive Surgical","Health Care"),
        ("SYK","Stryker","Health Care"),
        ("MDT","Medtronic","Health Care"),
        ("CVS","CVS Health","Health Care"),
        ("CI","Cigna","Health Care"),
        ("HUM","Humana","Health Care"),
        ("BIIB","Biogen","Health Care"),
        ("REGN","Regeneron","Health Care"),
        # ── Financials ──────────────────────────────────────────────────
        ("BRK-B","Berkshire","Financials"),
        ("JPM","JPMorgan","Financials"),
        ("V","Visa","Financials"),
        ("MA","Mastercard","Financials"),
        ("BAC","Bank of America","Financials"),
        ("WFC","Wells Fargo","Financials"),
        ("GS","Goldman Sachs","Financials"),
        ("MS","Morgan Stanley","Financials"),
        ("C","Citigroup","Financials"),
        ("AXP","American Express","Financials"),
        ("BLK","BlackRock","Financials"),
        ("SCHW","Charles Schwab","Financials"),
        ("CB","Chubb","Financials"),
        ("MMC","Marsh & McLennan","Financials"),
        ("PGR","Progressive","Financials"),
        ("USB","US Bancorp","Financials"),
        ("TFC","Truist Financial","Financials"),
        ("COF","Capital One","Financials"),
        ("ICE","Intercontinental Exchange","Financials"),
        ("CME","CME Group","Financials"),
        # ── Industrials ─────────────────────────────────────────────────
        ("GE","GE Aerospace","Industrials"),
        ("CAT","Caterpillar","Industrials"),
        ("BA","Boeing","Industrials"),
        ("RTX","Raytheon","Industrials"),
        ("HON","Honeywell","Industrials"),
        ("UPS","UPS","Industrials"),
        ("FDX","FedEx","Industrials"),
        ("DE","John Deere","Industrials"),
        ("LMT","Lockheed Martin","Industrials"),
        ("NOC","Northrop Grumman","Industrials"),
        ("GD","General Dynamics","Industrials"),
        ("MMM","3M","Industrials"),
        ("EMR","Emerson Electric","Industrials"),
        ("ITW","Illinois Tool Works","Industrials"),
        ("ETN","Eaton","Industrials"),
        ("PH","Parker Hannifin","Industrials"),
        ("CSX","CSX Corp","Industrials"),
        ("UNP","Union Pacific","Industrials"),
        ("NSC","Norfolk Southern","Industrials"),
        # ── Energy ──────────────────────────────────────────────────────
        ("XOM","ExxonMobil","Energy"),
        ("CVX","Chevron","Energy"),
        ("COP","ConocoPhillips","Energy"),
        ("EOG","EOG Resources","Energy"),
        ("SLB","Schlumberger","Energy"),
        ("MPC","Marathon Petroleum","Energy"),
        ("VLO","Valero Energy","Energy"),
        ("PSX","Phillips 66","Energy"),
        ("OXY","Occidental","Energy"),
        ("PXD","Pioneer Natural","Energy"),
        ("HAL","Halliburton","Energy"),
        ("DVN","Devon Energy","Energy"),
        ("FANG","Diamondback Energy","Energy"),
        ("UEC","Uranium Energy","Energy"),
        ("CCJ","Cameco","Energy"),
        ("URA","Global X Uranium ETF","Energy"),
        # ── Materials ───────────────────────────────────────────────────
        ("LIN","Linde","Materials"),
        ("SHW","Sherwin-Williams","Materials"),
        ("FCX","Freeport-McMoRan","Materials"),
        ("NUE","Nucor","Materials"),
        ("NEM","Newmont","Materials"),
        ("APD","Air Products","Materials"),
        ("ECL","Ecolab","Materials"),
        ("DD","DuPont","Materials"),
        ("PPG","PPG Industries","Materials"),
        ("ALB","Albemarle","Materials"),
        ("CF","CF Industries","Materials"),
        ("MOS","Mosaic","Materials"),
        # ── Utilities ───────────────────────────────────────────────────
        ("NEE","NextEra Energy","Utilities"),
        ("DUK","Duke Energy","Utilities"),
        ("SO","Southern Company","Utilities"),
        ("D","Dominion Energy","Utilities"),
        ("AEP","American Electric Power","Utilities"),
        ("EXC","Exelon","Utilities"),
        ("SRE","Sempra","Utilities"),
        ("XEL","Xcel Energy","Utilities"),
        ("PCG","PG&E","Utilities"),
        ("WEC","WEC Energy","Utilities"),
        # ── Real Estate ─────────────────────────────────────────────────
        ("PLD","Prologis","Real Estate"),
        ("SPG","Simon Property","Real Estate"),
        ("AMT","American Tower","Real Estate"),
        ("EQIX","Equinix","Real Estate"),
        ("CCI","Crown Castle","Real Estate"),
        ("PSA","Public Storage","Real Estate"),
        ("O","Realty Income","Real Estate"),
        ("DLR","Digital Realty","Real Estate"),
        ("WELL","Welltower","Real Estate"),
        ("VTR","Ventas","Real Estate"),
        # ── Information Technology (ek) ──────────────────────────────────
        ("MRVL","Marvell Tech","Information Technology"),
        ("ON","ON Semiconductor","Information Technology"),
        ("STX","Seagate","Information Technology"),
        ("WDC","Western Digital","Information Technology"),
        ("GEN","Gen Digital","Information Technology"),
        ("CTSH","Cognizant","Information Technology"),
        ("INFY","Infosys","Information Technology"),
        ("WIT","Wipro","Information Technology"),
        ("EPAM","EPAM Systems","Information Technology"),
        ("SNOW","Snowflake","Information Technology"),
        ("DDOG","Datadog","Information Technology"),
        ("NET","Cloudflare","Information Technology"),
        ("MDB","MongoDB","Information Technology"),
        ("TEAM","Atlassian","Information Technology"),
        ("ZS","Zscaler","Information Technology"),
        ("OKTA","Okta","Information Technology"),
        ("HUBS","HubSpot","Information Technology"),
        ("SHOP","Shopify","Information Technology"),
        # ── Communication Services (ek) ──────────────────────────────────
        ("SNAP","Snap","Communication Services"),
        ("PINS","Pinterest","Communication Services"),
        ("RBLX","Roblox","Communication Services"),
        ("MTCH","Match Group","Communication Services"),
        ("WBD","Warner Bros Discovery","Communication Services"),
        ("PARA","Paramount","Communication Services"),
        # ── Consumer Discretionary (ek) ──────────────────────────────────
        ("CPRT","Copart","Consumer Discretionary"),
        ("ORLY","O'Reilly Auto","Consumer Discretionary"),
        ("AZO","AutoZone","Consumer Discretionary"),
        ("BBY","Best Buy","Consumer Discretionary"),
        ("DRI","Darden Restaurants","Consumer Discretionary"),
        ("YUM","Yum! Brands","Consumer Discretionary"),
        ("CMG","Chipotle","Consumer Discretionary"),
        ("DECK","Deckers Outdoor","Consumer Discretionary"),
        ("RH","RH","Consumer Discretionary"),
        ("W","Wayfair","Consumer Discretionary"),
        # ── Consumer Staples (ek) ───────────────────────────────────────
        ("STZ","Constellation Brands","Consumer Staples"),
        ("TAP","Molson Coors","Consumer Staples"),
        ("CHD","Church & Dwight","Consumer Staples"),
        ("CLX","Clorox","Consumer Staples"),
        ("K","Kellogg","Consumer Staples"),
        ("CPB","Campbell Soup","Consumer Staples"),
        # ── Health Care (ek) ────────────────────────────────────────────
        ("ZBH","Zimmer Biomet","Health Care"),
        ("BAX","Baxter International","Health Care"),
        ("BDX","Becton Dickinson","Health Care"),
        ("EW","Edwards Lifesciences","Health Care"),
        ("HOLX","Hologic","Health Care"),
        ("IDXX","IDEXX Laboratories","Health Care"),
        ("IQV","IQVIA","Health Care"),
        ("LH","LabCorp","Health Care"),
        ("MCK","McKesson","Health Care"),
        ("MRNA","Moderna","Health Care"),
        ("VTRS","Viatris","Health Care"),
        ("VRTX","Vertex Pharma","Health Care"),
        ("PFE","Pfizer","Health Care"),
        ("AZN","AstraZeneca","Health Care"),
        ("NVO","Novo Nordisk","Health Care"),
        # ── Financials (ek) ─────────────────────────────────────────────
        ("ALL","Allstate","Financials"),
        ("MET","MetLife","Financials"),
        ("PRU","Prudential","Financials"),
        ("AFL","Aflac","Financials"),
        ("HIG","Hartford Financial","Financials"),
        ("TRV","Travelers","Financials"),
        ("BK","BNY Mellon","Financials"),
        ("STT","State Street","Financials"),
        ("NTRS","Northern Trust","Financials"),
        ("RJF","Raymond James","Financials"),
        ("AMP","Ameriprise","Financials"),
        ("TROW","T. Rowe Price","Financials"),
        ("WRB","W.R. Berkley","Financials"),
        ("RE","Everest Re","Financials"),
        # ── Industrials (ek) ────────────────────────────────────────────
        ("GWW","Grainger","Industrials"),
        ("CARR","Carrier Global","Industrials"),
        ("OTIS","Otis Worldwide","Industrials"),
        ("ROK","Rockwell Automation","Industrials"),
        ("AME","AMETEK","Industrials"),
        ("XYL","Xylem","Industrials"),
        ("LDOS","Leidos","Industrials"),
        ("SAIC","SAIC","Industrials"),
        ("VRSK","Verisk Analytics","Industrials"),
        ("CSGP","CoStar Group","Industrials"),
        ("BR","Broadridge Financial","Industrials"),
        ("PAYX","Paychex","Industrials"),
        ("ADP","ADP","Industrials"),
        # ── Energy (ek) ─────────────────────────────────────────────────
        ("MRO","Marathon Oil","Energy"),
        ("APA","APA Corp","Energy"),
        ("CTRA","Coterra Energy","Energy"),
        ("PR","Permian Resources","Energy"),
        ("LNG","Cheniere Energy","Energy"),
        ("KMI","Kinder Morgan","Energy"),
        ("WMB","Williams Companies","Energy"),
        ("OKE","ONEOK","Energy"),
        # ── Materials (ek) ──────────────────────────────────────────────
        ("IP","International Paper","Materials"),
        ("PKG","Packaging Corp","Materials"),
        ("RPM","RPM International","Materials"),
        ("CE","Celanese","Materials"),
        ("EMN","Eastman Chemical","Materials"),
        ("HUN","Huntsman","Materials"),
        ("ATI","ATI Inc","Materials"),
        ("RS","Reliance Steel","Materials"),
        # ── Utilities (ek) ──────────────────────────────────────────────
        ("ES","Eversource","Utilities"),
        ("ETR","Entergy","Utilities"),
        ("EIX","Edison International","Utilities"),
        ("AEE","Ameren","Utilities"),
        ("CMS","CMS Energy","Utilities"),
        ("LNT","Alliant Energy","Utilities"),
        # ── Real Estate (ek) ────────────────────────────────────────────
        ("EQR","Equity Residential","Real Estate"),
        ("AVB","AvalonBay","Real Estate"),
        ("MAA","Mid-America Apartment","Real Estate"),
        ("IRM","Iron Mountain","Real Estate"),
        ("CBRE","CBRE Group","Real Estate"),
    ]
    df = pd.DataFrame(rows, columns=["Symbol", "Name", "Sector"])
    df["SubIndustry"] = df["Sector"]
    return df


if __name__ == "__main__":
    df = fetch_sp500_tickers()
    print(f"Toplam: {len(df)} hisse")
    print(df.groupby("Sector").size().sort_values(ascending=False))
