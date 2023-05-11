import cv2
from google.colab.patches import cv2_imshow
 
file=r'/content/img.png'
image=cv2.imread(file,0)

print(image.shape)
# cv2_imshow(image)
image = image[..., ::-1]
# getType(image)
im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


########

import layoutparser as lp
file=r'/content/img.png'
image=cv2.imread(file,0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.9],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})

layout = model.detect(image) # You need to load the image somewhere else, e.g., image = cv2.imread(...)
lp.draw_box(image, layout,)


#########

text_blocks = lp.Layout([b for b in layout if b.type=="Text" or b.type=="Title"])
figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
text_blocks = lp.Layout([b for b in text_blocks \
                   if not any(b.is_in(b_fig) for b_fig in figure_blocks)])
                   
#########


h, w = image.shape[:2]

left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

left_blocks = text_blocks.filter_by(left_interval, center=True)
# left_blocks.sort(key = lambda b:b.coordinates[1])

right_blocks = [b for b in text_blocks if b not in left_blocks]
# right_blocks.sort(key = lambda b:b.coordinates[1])

#  finally we combine the two list and add the index according to the order
text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
lp.draw_box(image, text_blocks,
            box_width=3, 
            show_element_id=False)
            

ocr_agent = lp.TesseractAgent(languages='eng') 
for block in text_blocks:
    segment_image = (block
                       .pad(left=5, right=5, top=500, bottom=5)
                       .crop_image(image))
        # we added a padding in each image segment  
        
    text = ocr_agent.detect(segment_image)
    block.set(text=text, inplace=True)
    

###########

for txt in text_blocks.get_texts():
    print(txt, end='\n---\n')
    s=txt

L=s.split("\n")
L[0]


###########


import xlsxwriter
 
workbook = xlsxwriter.Workbook('Example2.xlsx')
worksheet = workbook.add_worksheet()

row = 0
column = 0

for item in L :

    worksheet.write(row, column, item)

    row += 1

workbook.close()

