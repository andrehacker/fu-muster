function result=vectorAngle(a, b)
    tmp = dot(a,b)/(norm(a)*norm(b));
    result = acos(tmp)*(180/pi);
end 