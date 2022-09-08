 
import bpy
from bpy import context, data, ops

def render_images(path_obj, path_mat,path_background):
    """
    In Windows double backlslashes have to be used for the file paths 
    path_obj : path to obj data 
    path_mat : path to material png / jpg 
    path_background :  path to background HDRI file .exr 

    return None 
    Output : Images from a circle around the object with a given background and material for the surface 
    """
    #loading data and adding material to object 
    imported_object = bpy.ops.import_scene.obj(filepath=path_obj)
    obj_object = bpy.context.selected_objects[0]
    mat = bpy.data.materials.new(name="New_Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(path_mat)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    ob = context.view_layer.objects[1]
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat) 
    print('Imported name: ', obj_object.name)
    # Creating circe that allows the camera to move around the object 
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(5, 5, 5))
    bpy.ops.curve.primitive_bezier_circle_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0),               scale=(1, 1, 1))
    bpy.ops.transform.resize(   value=(8.09934, 8.09934, 8.09934), 
                            orient_type='GLOBAL', 
                            orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                            orient_matrix_type='GLOBAL', 
                            mirror=False, use_proportional_edit=False, 
                            proportional_edit_falloff='SMOOTH', 
                            proportional_size=1,
                            use_proportional_connected=False, 
                            use_proportional_projected=False)
    circle = bpy.context.object

    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 8.09934, 0), rotation=(0.0, 0.0, 0.0),      scale=(1, 1, 1))
    camera = bpy.context.object
    bpy.ops.object.constraint_add(type='TRACK_TO')
    bpy.context.object.constraints["Track To"].target = bpy.data.objects["Empty"]

    bpy.ops.object.select_all(action='DESELECT')
    circle.select_set(True)

    camera.select_set(True)
    bpy.context.view_layer.objects.active = circle  
    bpy.ops.object.parent_set(type='FOLLOW')
    #HDR Background 
    context = bpy.context
    scene = context.scene

    # get tree for scene 
    node_tree = scene.world.node_tree
    nodes = node_tree.nodes

    # clear
    nodes.clear()
    node_background = nodes.new(type='ShaderNodeBackground')

    #background texture 
    node_environment = nodes.new('ShaderNodeTexEnvironment')
    #load image a
    node_environment.image = bpy.data.images.load(path_background) # Relative path
    node_environment.location = -300,0

    # Add as background node 
    node_output = nodes.new(type='ShaderNodeOutputWorld')   
    node_output.location = 200,0

    # link everything 
    links = node_tree.links
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
    # Animation of Camera moving around the object 
    bpy.context.object.data.path_duration = 30
    bpy.context.scene.frame_end = 30
    bpy.context.scene.render.fps = 10


    ## Rendering 
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.camera = camera
    bpy.ops.render.render(animation=False, write_still=True)

if __name__ == "__main__":
    pass 
    
