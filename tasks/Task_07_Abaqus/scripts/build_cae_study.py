import argparse
import os
import sys
from pathlib import Path

import mesh
import regionToolset
from abaqus import Mdb, mdb
from abaqusConstants import ANALYSIS, DEFAULT, DEFORMABLE_BODY, FIELD, FINER, OFF, ON, QUAD, S3, S4R, STANDARD, STRUCTURED, THREE_D


def _discover_script_dir():
    candidates = []
    for hint in (globals().get('__file__'), sys.argv[0] if sys.argv else None):
        if not hint:
            continue
        path = Path(hint)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        candidates.append(path.parent if path.suffix else path)
    candidates.append(Path.cwd())
    candidates.append((Path.cwd() / 'tasks' / 'Task_07_Abaqus' / 'scripts').resolve())
    for candidate in candidates:
        if (candidate / 'task7_common.py').exists():
            return str(candidate)
    return str((Path.cwd() / 'tasks' / 'Task_07_Abaqus' / 'scripts').resolve())


SCRIPT_DIR = _discover_script_dir()
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from task7_common import load_json, refined_job_name


def _log_line(path, message):
    with open(path, 'a') as handle:
        handle.write(message + '\n')


def _parse_args():
    script_path = Path(SCRIPT_DIR) / 'build_cae_study.py'
    repo_root = script_path.resolve().parents[3]
    default_cases = repo_root / 'tasks' / 'Task_07_Abaqus' / 'candidates' / 'selected_cases.json'
    default_config = repo_root / 'tasks' / 'Task_07_Abaqus' / 'config' / 'study_config.json'
    default_output = repo_root / 'tasks' / 'Task_07_Abaqus' / 'results' / 'cae' / 'Task7_Montresor_WindStudy'

    argv = sys.argv[:]
    if '--' in argv:
        argv = argv[argv.index('--') + 1 :]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(description='Build a clean Abaqus/CAE Task 7 study.')
    parser.add_argument('--cases', default=str(default_cases))
    parser.add_argument('--config', default=str(default_config))
    parser.add_argument('--output-cae', default=str(default_output))
    return parser.parse_args(argv)


def _classify_edges(part, base_z, top_z, tol):
    circumferential = []
    meridional = []
    base_edges = []
    top_edges = []
    for edge in part.edges:
        vertices = edge.getVertices()
        if len(vertices) == 1:
            circumferential.append(edge)
            height_val = float(edge.pointOn[0][1])
            if abs(height_val - base_z) <= tol:
                base_edges.append(edge)
            if abs(height_val - top_z) <= tol:
                top_edges.append(edge)
        else:
            meridional.append(edge)
    return circumferential, meridional, base_edges, top_edges


def _create_native_shell_part(model, case, config):
    max_radius = max(float(value) for value in case['radii_m'])
    total_height = float(case['z_m'][-1] - case['z_m'][0])
    sheet_size = 4.0 * max(max_radius, total_height, 1.0)

    sketch = model.ConstrainedSketch(name='TowerProfile', sheetSize=sheet_size)
    sketch.ConstructionLine(point1=(0.0, -sheet_size), point2=(0.0, sheet_size))
    points = list(zip(case['radii_m'], case['z_m']))
    for start, end in zip(points[:-1], points[1:]):
        sketch.Line(point1=(float(start[0]), float(start[1])), point2=(float(end[0]), float(end[1])))

    part = model.Part(name='TowerShell', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseShellRevolve(sketch=sketch, angle=360.0, flipRevolveDirection=OFF)
    del model.sketches['TowerProfile']

    part.Set(faces=part.faces[:], name='FALL')
    tol = max(1.0e-6, total_height * 1.0e-6)
    circumferential, meridional, base_edges, top_edges = _classify_edges(part, float(case['z_m'][0]), float(case['z_m'][-1]), tol)
    if not base_edges:
        raise ValueError('Failed to identify base edge for {0}'.format(case['model_name']))
    base_edge = base_edges[0]
    part.Set(edges=part.edges[base_edge.index : base_edge.index + 1], name='EBASE')
    if top_edges:
        top_edge = top_edges[0]
        part.Set(edges=part.edges[top_edge.index : top_edge.index + 1], name='ETOP')

    refined_theta = int(config['mesh']['refined_circumferential_divisions'])
    refined_axial = int(config['mesh']['refined_axial_subdivisions_per_segment'])
    for edge in circumferential:
        part.seedEdgeByNumber(edges=(edge,), number=refined_theta, constraint=FINER)
    for edge in meridional:
        part.seedEdgeByNumber(edges=(edge,), number=refined_axial, constraint=FINER)

    # Prefer structured meshing per face; fall back to free meshing when mapping is not possible.
    for face in part.faces:
        try:
            part.setMeshControls(regions=(face,), elemShape=QUAD, technique=STRUCTURED)
        except Exception:
            part.setMeshControls(regions=(face,), elemShape=QUAD)
    elem_type_quad = mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD)
    elem_type_tri = mesh.ElemType(elemCode=S3, elemLibrary=STANDARD)
    part.setElementType(regions=(part.faces[:],), elemTypes=(elem_type_quad, elem_type_tri))
    part.generateMesh()
    return part


def _populate_model(case, config, model=None):
    model_name = case['model_name']
    if model is None:
        model = mdb.Model(name=model_name)

    part = _create_native_shell_part(model, case, config)

    material_name = config['material']['name']
    section_name = 'TowerSection'
    model.Material(name=material_name)
    model.materials[material_name].Elastic(table=((float(config['material']['youngs_modulus_pa']), float(config['material']['poisson_ratio'])),))
    model.materials[material_name].Density(table=((float(config['material']['density_kg_m3']),),))
    model.HomogeneousShellSection(name=section_name, material=material_name, thickness=float(config['shell']['thickness_m']))
    part.SectionAssignment(region=part.sets['FALL'], sectionName=section_name)

    assembly = model.rootAssembly
    instance = assembly.Instance(name='Tower-1', part=part, dependent=ON)

    model.StaticStep(name='STATIC_WIND', previous='Initial', nlgeom=OFF)
    model.BuckleStep(name='BUCKLING', previous='STATIC_WIND', numEigen=int(config['buckling']['num_eigenvalues']))

    model.ExpressionField(name='WindCp', expression=config['wind']['pressure_field_expression'])
    gravity = config['gravity']
    model.Gravity(
        name='SelfWeight',
        createStepName='STATIC_WIND',
        comp1=float(gravity['direction'][0]) * float(gravity['acceleration_m_s2']),
        comp2=float(gravity['direction'][1]) * float(gravity['acceleration_m_s2']),
        comp3=float(gravity['direction'][2]) * float(gravity['acceleration_m_s2']),
    )

    q_ref = 0.5 * float(config['wind']['air_density_kg_m3']) * float(config['wind']['reference_speed_m_s']) ** 2
    wind_region = regionToolset.Region(side1Faces=instance.faces)
    model.Pressure(
        name='Wind',
        createStepName='STATIC_WIND',
        region=wind_region,
        magnitude=-q_ref,
        distributionType=FIELD,
        field='WindCp',
    )

    model.DisplacementBC(
        name='FixBase',
        createStepName='Initial',
        region=instance.sets['EBASE'],
        u1=0.0,
        u2=0.0,
        u3=0.0,
        ur1=0.0,
        ur2=0.0,
        ur3=0.0,
    )
    return model_name


def main():
    args = _parse_args()
    cases = load_json(Path(args.cases))
    config = load_json(Path(args.config))
    output_cae = os.path.abspath(args.output_cae)
    if output_cae.lower().endswith('.cae'):
        output_cae = output_cae[:-4]
    output_dir = os.path.dirname(output_cae)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_path = os.path.join(output_dir, 'build_cae_study.log')
    if os.path.exists(log_path):
        os.remove(log_path)

    cpu_count = int(config.get('jobs', {}).get('cpus', 1))
    Mdb()
    first_case = True
    for case in cases:
        model_name = case['model_name']
        if first_case and 'Model-1' in mdb.models and len(mdb.models.keys()) == 1:
            try:
                mdb.models.changeKey(fromName='Model-1', toName=model_name)
                model = mdb.models[model_name]
            except Exception as exc:
                _log_line(log_path, 'reuse_default_model_failed={0}'.format(exc))
                model = mdb.Model(name=model_name)
        else:
            model = mdb.Model(name=model_name)

        model_name = _populate_model(case, config, model=model)
        job_name = refined_job_name(case)
        mdb.Job(
            name=job_name,
            model=model_name,
            description=case['label'],
            type=ANALYSIS,
            multiprocessingMode=DEFAULT,
            numCpus=cpu_count,
            numDomains=cpu_count,
        )
        _log_line(log_path, 'created_model_and_job={0} -> {1}'.format(model_name, job_name))
        first_case = False

    target_path = output_cae + '.cae'
    if os.path.exists(target_path):
        try:
            os.remove(target_path)
        except Exception as exc:
            raise RuntimeError('Target CAE path is locked or unavailable: {0}'.format(exc))

    mdb.saveAs(pathName=output_cae)
    _log_line(log_path, 'saved={0}'.format(target_path))
    print('Saved Task 7 CAE study to: {0}'.format(target_path))


if __name__ == '__main__':
    main()
