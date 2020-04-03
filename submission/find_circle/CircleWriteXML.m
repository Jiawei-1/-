function CircleWriteXML(filename,radius,x,y)
%�����ĵ������ڵ�
% filename='C:\Users\gjw\Desktop\dimension.xml'
if (exist(filename,'file')==0)
doc = com.mathworks.xml.XMLUtils.createDocument('dimension');
%���ڵ�
else
doc=xmlread(filename);        
end
docRootNode = doc.getDocumentElement;
%������һ��ע����Ԫ��
circle = doc.createElement('circle');
circle.setAttribute('radius',num2str(radius));
point=doc.createElement('point');
point.setAttribute('x',num2str(x));
point.setAttribute('y',num2str(y));
circle.appendChild(point);
%��ע��Ԫ�ظ��ڸ��ڵ��������
docRootNode    = docRootNode.appendChild(circle);
xmlwrite(filename,doc);
end