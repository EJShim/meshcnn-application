import vtk
import torch


renderer = vtk.vtkRenderer()
renderer.SetBackground(0.1, 0.5, 0.5)
renWin = vtk.vtkRenderWindow()
renWin.SetSize(500, 500)
renWin.AddRenderer(renderer)

interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.SetInteractorStyle(interactorStyle)
iren.Initialize()



if __name__ == "__main__":
    
    reader = vtk.vtkOBJReader()
    reader.SetFileName("data/human_seg/shrec__2_0.obj")
    reader.Update()


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(reader.GetOutput())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)


    renderer.AddActor(actor)

    renderer.ResetCamera()
    renWin.Render()


    #Initialize Torch
    opt = dict(arch='meshunet', batch_size=12, checkpoints_dir='./checkpoints', dataroot='datasets/human_seg', dataset_mode='segmentation', export_folder='./checkpoints\\human_seg\\meshes', fc_n=100, gpu_ids=[0], init_gain=0.02, init_type='normal', is_train=False, max_dataset_size=inf, name='human_seg', ncf=[32, 64, 128, 256], ninput_edges=2280, norm='batch', num_aug=1, num_groups=16, num_threads=3, phase='test', pool_res=[1800, 1350, 600], resblocks=3, results_dir='./results/', seed=None, serial_batches=True, which_epoch='latest')

    iren.Start()

