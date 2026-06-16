import sqlite3
import pandas as pd
import re
from pathlib import Path

# ACX = "/home/adios/dropbox/adios-campaign-store/kstar.acx"

# Define five control parameters (i.e., inputs)
#   ip:     plasma current [in kA],
#   ncore:  core electron (roughly psin=0.95) density [in m^-3]
#   pinj:   total injection power [in MW]
#   fz:     impurity fraction
#   diff:   diffusion coefficient scaling factor

# values_ip = [300, 500, 600, 700, 800]
# values_ncore = []
# values_pinj = [
#    0.5, 0.6, 0.75, 1.0, 1.0714285714285714, 1.0833333333333335, 1.25, 1.5, 1.5428571428571427,  1.6666666666666667,
#    1.75, 2.0, 2.0142857142857142, 2.25, 2.4857142857142858, 2.5, 2.75, 2.8333333333333335, 2.9571428571428573,
#    3.0, 3.25, 3.416666666666667, 3.4285714285714284, 3.5, 3.75, 3.9, 4.0
# ]
# values_fz = []
# values_diff = [
#    0.6, 0.65, 0.7999999999999999, 0.8666666666666667, 1.0, 1.0833333333333333, 1.2, 1.2999999999999998, 1.4,
#    1.5166666666666666, 1.5999999999999999, 1.7333333333333334, 1.7999999999999998, 1.95, 2.0
# ]


def GetParameters(
    ACX: str,
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    # num = r"([0-9]+(?:\.[0-9]+)?)"
    num = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    pattern1 = rf"Ip([0-9]+)_p{num}_d{num}(?:\.|$)"
    pattern2 = rf"n{num}/f{num}(?:/|$)"
    Ip_list, p_list, d_list, n_list, f_list = [], [], [], [], []
    con = sqlite3.connect(ACX)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    nrows, mrows = 0, 0
    res = cur.execute(
        "SELECT a.name, d.name FROM archives as a JOIN datasets as d ON d.archiveid = a.rowid WHERE d.name LIKE '%/images';"
    )
    for row in res:
        nrows += 1
        m1 = re.search(pattern1, row[0])
        m2 = re.search(pattern2, row[1])
        if m1 and m2:
            mrows += 1
            Ip = int(m1.group(1))
            p = float(m1.group(2))
            d = float(m1.group(3))
            n = float(m2.group(1))
            f = float(m2.group(2))
            Ip_list.append(Ip)
            p_list.append(p)
            d_list.append(d)
            n_list.append(n)
            f_list.append(f)
        else:
            print(f"{row[0]} | {row[1]}")

    cur.close()
    con.close()
    print(f"rows = {nrows}, valid = {mrows}")
    return Ip_list, p_list, d_list, n_list, f_list


def GetDataFrame(
    Ips: list[int], ps: list[float], ds: list[float], ns: list[float], fs: list[float]
) -> pd.DataFrame:
    df = pd.DataFrame({"Ip": Ips, "p": ps, "d": ds, "n": ns, "f": fs})
    return df


def select_validation_dir(base: Path):
    if not base.exists() or not base.is_dir():
        print(f"Directory '{base}' does not exist.")
        return None

    # Collect valid directories and parse names
    options = []
    for p in base.iterdir():
        if p.is_dir():
            try:
                M_str, r_str = p.name.split("_")
                M, r = int(M_str), int(r_str)
                options.append((M, r, p))
            except ValueError:
                # Skip directories that don't match pattern
                continue

    if not options:
        print("No valid directories found.")
        return None

    # Sort for nicer display
    options.sort()

    # Show choices
    print("Available validation sets:")
    for i, (M, r, _) in enumerate(options):
        print(f"{i}: M={M}, r={r}")

    # Ask user
    while True:
        try:
            choice = input("Select an index: ").strip()
            try:
                idx = int(choice)
                if 0 <= idx < len(options):
                    return options[idx][2]  # return Path
            except ValueError:
                pass
            print("Invalid selection, try again.")
        except (KeyboardInterrupt, EOFError):
            print("\nInput aborted.")
            return None


def ask_an_integer(quote: str) -> int:
    # Ask user
    while True:
        try:
            choice = input(quote).strip()
            try:
                n = int(choice)
                if n >= 0:
                    return n
            except ValueError:
                pass
            print("Enter a non-negative integer please.")
        except (KeyboardInterrupt, EOFError):
            print("\nInput aborted.")
            return -1
