function CircleWriteXML(filename,radius,x,y)
%建立文档及根节点
% filename='C:\Users\gjw\Desktop\dimension.xml'
if (exist(filename,'file')==0)
doc = com.mathworks.xml.XMLUtils.createDocument('dimension');
%根节点
else
doc=xmlread(filename);        
end
docRootNode = doc.getDocumentElement;
%建立第一个注释子元素
circle = doc.createElement('circle');
circle.setAttribute('radius',num2str(radius));
point=doc.createElement('point');
point.setAttribute('x',num2str(x));
point.setAttribute('y',num2str(y));
circle.appendChild(point);
%将注释元素附在根节点的子项上
docRootNode    = docRootNode.appendChild(circle);
xmlwrite(filename,doc);
end