from pymol import cmd, stored


def visualize(pdbid, method):
    assert method in ["gradient", "atomic"]

    # Clear everything
    cmd.reinitialize("everything")

    ligname = f"{pdbid}_ligand_{method}"
    recname = f"{pdbid}_protein_{method}"

    ligfile = ligname + ".pdb"
    recfile = recname + ".pdb"

    cmd.load(ligfile, "lig")
    cmd.load(recfile, "rec")

    # Hide everything
    cmd.hide("all")

    # Show ligand
    cmd.show("sticks", "lig")
    # cmd.show("spheres", "lig")
    # cmd.set("sphere_scale", 0.1, "lig")
    # cmd.set("stick_radius", 0.1, "lig")

    # Show protein
    cmd.select("residues", "byres (rec within 3.5 of lig)")
    cmd.show("cartoon", "rec")
    cmd.color("grey", "rec")
    cmd.show("sticks", "residues")

    # Extract max/min b-factors
    stored.bfs = []
    cmd.iterate("all", "stored.bfs.append(b)")  # Selection

    Bmax = max(max(stored.bfs), abs(min(stored.bfs)))

    cmd.spectrum("b", "orange_white_blue", "lig", -Bmax, Bmax)
    cmd.spectrum("b", "red_white_green", "residues", -Bmax, Bmax)

    cmd.remove("(hydro)")
    cmd.remove("solvent")

    cmd.center("lig")
    cmd.zoom("lig", 5)


cmd.extend("visualize", visualize)
