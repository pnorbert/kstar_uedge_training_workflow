import re
from pathlib import Path

import numpy as np
from adios2 import Adios, FileReader, Stream

# Define five control parameters (i.e., inputs)
#   ip:     plasma current [in kA],
#   ncore:  core electron (roughly psin=0.95) density [in m^-3]
#   pinj:   total injection power [in MW]
#   fz:     impurity fraction
#   diff:   diffusion coefficient scaling factor
ip_list, ncore_list, pinj_list, fz_list, diff_list = [], [], [], [], []

# Define diagnostics (i.e., outputs)
#   neu:    electron density profile at outer mid-plane (or upstream) [in m^-3]
#   teu:    electron temperature profile at outer mid-plane [in eV]
#   ter:    electron temperature profile at outer divertor target [in eV]
#   tel:    electron temperature profile at inner divertor target [in eV]
#   jr:     ion saturation current density profile at outer divertor target [in A/m^2]
#   jl:     ion saturation current density profile at inner divertor target [in A/m^2]
#   qtr:    total paralle heat flux profile at outer divertor target [in W/m^2]
#   qtl:    total paralle heat flux profile at inner divertor target [in W/m^2]
#   rads:   radiation information: radiated power fraction, divertor radiation fraction, peak radiation location
neu_list, teu_list, ter_list, tel_list = [], [], [], []
jr_list, jl_list, qtr_list, qtl_list, rads_list = [], [], [], [], []

ee = 1.60217663e-19  # Elementray charge [in C]

pattern_campaign = re.compile(r".*Ip(\d+)_p(\d+\.\d+)_d(\d+\.\d+)\.aca")
pattern_varpath = re.compile(r"n(\d+\.\d+)/f([^\/]*)/images/rads")


def read_one_campaign(campaign: Path, ip: int, pinj: float, diff: float, nf_pairs: list[tuple]) -> int:
    if not nf_pairs:
        return 0

    case_count = 0
    image_paths = [f"n{ncore}/f{fz}/images" for ncore, fz in nf_pairs]

    adios = Adios()
    io = adios.declare_io("campaign-reader")
    io.set_parameter("include-dataset", ";".join(re.escape(path) for path in image_paths))

    with FileReader(io, str(campaign)) as f:
        all_vars = f.available_variables()
        for (ncore, fz), imgpath in zip(nf_pairs, image_paths, strict=True):
            # print(f"  {imgpath}")

            # Check for required dataset
            if f"{imgpath}/qtl_new" not in all_vars:
                print(f"    ERROR: Not found {imgpath}/qtl_new, skip case")
                continue  # Skip if required data is missing

            # Collect input parameters
            ip_list.append(float(ip))
            ncore_list.append(float(ncore))
            pinj_list.append(float(pinj))
            fz_list.append(float(fz))
            diff_list.append(float(diff))

            # Collect output diagnostics
            # Read all required data in deferred mode to allow ADIOS multithread the reading
            # from remote storage
            # Ion saturation current
            if f"{imgpath}/jsatr" in all_vars:
                jr = f.read(f"{imgpath}/jsatr", defer_read=True)
                jl = f.read(f"{imgpath}/jsatl", defer_read=True)
            elif f"{imgpath}/jl" in all_vars:
                print(f"    WARNING: No jsatl/jsatr, using jr/jl from {campaign} / {imgpath}")
                jr = f.read(f"{imgpath}/jr", defer_read=True)
                jl = f.read(f"{imgpath}/jl", defer_read=True)
            else:
                print("    ERROR: Missing ion saturation current info. skip case.")
                continue
            ### neu = f.read(f"{imgpath}/ni", start=[33, 1, 0], count=[1,-1,1], defer_read=True)
            vni = f.inquire_variable(f"{imgpath}/ni")
            if vni is None:
                continue
            vni.set_selection([[33, 1, 0], [1, vni.shape()[1] - 2, 1]])
            neu = f.read(vni, defer_read=True)

            te = f.read(f"{imgpath}/te", defer_read=True)
            # Heat flux
            qtr_new = f.read(
                f"{imgpath}/qtr_new", defer_read=True
            )  # qtr_new adds radiation and atomic process contribution
            qtl_new = f.read(f"{imgpath}/qtl_new", defer_read=True)
            qradhl = f.read(f"{imgpath}/qradhl", defer_read=True)
            qradzl = f.read(f"{imgpath}/qradzl", defer_read=True)
            # Radiation info
            rads = f.read(f"{imgpath}/rads", defer_read=True)

            # complete all read operations now
            f.read_complete()

            neu_list.append(np.squeeze(neu))
            teu_list.append(te[33, 1:-1])
            ter_list.append(te[-1, 1:-1])
            tel_list.append(te[1, 1:-1])
            jr_list.append(jr[1:-1])
            jl_list.append(jl[1:-1])
            # Heat flux
            qtr_list.append(qtr_new[1:-1])  # qtr_new adds radiation and atomic process contribution
            qtl_tmp = qtl_new[:] - 2.0 * (qradhl[:] + qradzl[:])
            qtl_list.append(qtl_tmp[1:-1])
            # Radiation info
            rads_list.append(rads[:])
            case_count += 1
    return case_count


def combine_data(output: Path | None = None, append: bool = True):
    # Convert lists to numpy arrays and reshape as needed
    ip = np.array(ip_list)
    ncore = np.array(ncore_list)
    pinj = 2.0 * np.array(pinj_list)  # pinj is half of total injection power in UEDGE
    fz = np.array(fz_list)
    diff = np.array(diff_list)

    neu = np.array(neu_list) / 1e19  # [1e19/m^3]
    teu = np.array(teu_list) / ee  # [eV]
    ter = np.array(ter_list) / ee  # [eV]
    tel = np.array(tel_list) / ee  # [eV]
    jr = np.array(jr_list)
    jl = -np.array(jl_list)  # Reverse sign for positivity
    qtr = np.array(qtr_list)
    qtl = -np.array(qtl_list)  # Reverse sign for positivity
    rads = np.array(rads_list)

    if output is not None:
        if append:
            mode = "a"
        else:
            mode = "w"
        with Stream(str(output), mode) as f:
            f.write("ip", ip)
            f.write("ncore", ncore)
            f.write("pinj", pinj)
            f.write("fz", fz)
            f.write("diff", diff)
            f.write("neu", neu)
            f.write("teu", teu)
            f.write("ter", ter)
            f.write("tel", tel)
            f.write("jr", jr)
            f.write("jl", jl)
            f.write("qtr", qtr)
            f.write("qtl", qtl)
            f.write("rads", rads)
            f.write("nsamples", ip.size)

    return ip, ncore, pinj, fz, diff, neu, teu, ter, tel, jr, qtr, qtl, rads


def load_data(output: Path):
    variable_names = ("ip", "ncore", "pinj", "fz", "diff", "neu", "teu", "ter", "tel", "jr", "qtr", "qtl", "rads")
    values_by_name = {name: [] for name in variable_names}

    with Stream(str(output), "r") as stream:
        for _ in stream.steps():
            for name in variable_names:
                values_by_name[name].append(np.array(stream.read(name), copy=True))

    if not values_by_name["ip"]:
        raise ValueError(f"No steps found in ADIOS2 file {output}")

    ip, ncore, pinj, fz, diff, neu, teu, ter, tel, jr, qtr, qtl, rads = (
        np.concatenate(values_by_name[name], axis=0) for name in variable_names
    )

    return ip, ncore, pinj, fz, diff, neu, teu, ter, tel, jr, qtr, qtl, rads
