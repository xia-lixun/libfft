			--------------------------
			--fft configuration file--
			--lixun.xia@outlook.com --
			--     2016-09-08       --
			-- All rights reserved. --
			--------------------------


--1. complex number in x-y coordinate
--   x+y*i
--
function complex_rectan(x, y)
	 local z = {re=x, im=y}
	 return z
end


--2. complex number in polar coordinate
--   a*exp(theta*i)
--
function complex_polar(a, theta)
	 local z = {re=a*math.cos(theta), im=a*math.sin(theta)}
	 return z
end


--3. complex number addition
--
function add(za, zb)
	 local z = {re=za.re+zb.re, im=za.im+zb.im}
	 return z
end


--4. complex number subtraction
--
function sub(za, zb)
	 local z = {re=za.re-zb.re, im=za.im-zb.im}
	 return z
end


--5. complex number multiplication
--
function mul(za, zb)
	 local z = {re=(za.re * zb.re) - (za.im * zb.im), 
	       	    im=(za.re * zb.im) + (za.im * zb.re)}
	 return z
end


--6. complex number exponential
--   exp(x+y*i) = exp(x) * [cos(y) + sin(y)*i]
function exp(z)
	 local za = {re=math.exp(z.re) * math.cos(z.im),
	             im=math.exp(z.re) * math.sin(z.im)}
	 return za
end


-------------------------------------------------------------------
--function  : init twiddle factors for conjugate-pair fft
--n	    : n-point fft
-------------------------------------------------------------------
function fft_int(n)
	 local wnk = {}
	 
	 local n4 = n / 4
	 for k = 0, n4-1, 1 do
	     local phi = complex_rectan(0, -2 * math.pi * k / n)
	     wnk[k] = exp(phi)
	 end

	 return wnk
end



is_mid_child = true
_cnt4 = 0
_ncat = 0
-------------------------------------------------------------------
--function  : generates recursive fft
--px	    : reference to input complex vector
--leaf	    : switch toggling leaf/branch states
-------------------------------------------------------------------
function fft(px, tree, trace, leaf, leaf_code, leaf_param)
	 
	 local n = #px + 1
	 local y = {}
	 for k = 0, n-1, 1 do y[k] = complex_rectan(0,0) end

	 --
	 if n==1 then

		if leaf then
		   if is_mid_child then 
		      is_mid_child = false
		      if next(tree[trace]) ~= nil then table.insert(leaf_param, tree[trace][0].re) end
		   else
		      is_mid_child = true
		      if next(tree[trace]) ~= nil then table.insert(leaf_param, tree[trace][0].re) end
		      _ncat = _ncat + 1
		      _cnt4 = _cnt4 + 1
		      if _cnt4 == 4 then _cnt4 = 0 table.insert(leaf_code, _ncat) _ncat = 0  end
		   end
		end

	    	y[0] = px[0]
	    	return y

	 --
	 elseif n==2 then 

	 	if leaf then
		   if next(tree[trace]) ~= nil then table.insert(leaf_param, tree[trace][0].re) end
		   if next(tree[trace]) ~= nil then table.insert(leaf_param, tree[trace][1].re) end
		   _ncat = _ncat + 2
		   _cnt4 = _cnt4 + 1
		   if _cnt4 == 4 then _cnt4 = 0 table.insert(leaf_code, _ncat) _ncat = 0  end
		end

	 	y[0] = add(px[0], px[1])  
		y[1] = sub(px[0], px[1])
	 	return y

	 --
	 else

		if not leaf then
		   --place holder for higher order codelets
		end


		local pxu = {}
		local pxz = {}
		local pxz_ = {}

		for k = 0, n/2-1, 1 do  pxu[k] = px[k*2]  end
		for k = 0, n/4-1, 1 do  pxz[k] = px[k*4+1]  end
		pxz_[0] = px[n-1]
		for k = 1, n/4-1, 1 do  pxz_[k] = px[k*4-1]  end

		local u = fft(pxu, tree, trace * 3 + 1, leaf, leaf_code, leaf_param)
		local z = fft(pxz, tree, trace * 3 + 2, leaf, leaf_code, leaf_param)
		local z_ = fft(pxz_, tree, trace * 3 + 3, leaf, leaf_code, leaf_param)

		for k = 0, n/4-1, 1 do
		      	local phi = complex_rectan(0, -2 * math.pi * k / n)
	     		local phi_ = complex_rectan(0, 2 * math.pi * k / n)
			local wnk = exp(phi)
			local wnk_ = exp(phi_)
			local complex_i = complex_rectan(0, 1)

			local a = mul(wnk, z[k])
			local b = mul(wnk_, z_[k])
			local apb = add(a, b)
			local asb = sub(a, b)
			local asbi = mul(complex_i, asb)

			y[k] = add(u[k], apb)
			y[k + n/2] = sub(u[k], apb)
			y[k + n/4] = sub(u[k + n/4], asbi)
			y[k + 3*n/4] = add(u[k + n/4], asbi) 

		end
		return y
	 end

--function end
end








-------------------------------------------------------------------
--function	: generates recursive tree of the fft, child
--heap    	: heap tree
--parent_node 	: paranet node
-------------------------------------------------------------------
function generate_child(heap, parent_node)

	 local parent_n = #(heap[parent_node]) + 1
	 
	 local child_l = {}
	 local child_m = {}
	 local child_r = {}

	 if parent_n >= 4 then
	    	     for k = 0, parent_n/2-1, 1 do  child_l[k] = heap[parent_node][k*2]  end
		     for k = 0, parent_n/4-1, 1 do  child_m[k] = heap[parent_node][k*4+1]  end
		     child_r[0] = heap[parent_node][parent_n - 1]
		     for k = 1, parent_n/4-1, 1 do  child_r[k] = heap[parent_node][k*4-1]  end
	 end

	 heap[#heap + 1] = child_l
	 heap[#heap + 1] = child_m
	 heap[#heap + 1] = child_r

	 return
end



--px	: reference to input complex array
function generate_tree(px)

	 local logn = math.floor( math.log(#px+1) / math.log(2) + 0.5 )
	 local tree = {}
	 local node_n = #tree

	 tree[0] = px
	 local delta_node_n = (#tree + 1) - node_n
	 node_n = (#tree + 1)

	 local parent_node = 0
	 for i = 0, logn-2, 1 do
	     for j = 0, delta_node_n-1, 1 do
	     	 generate_child(tree, parent_node + j)
	     end
	     parent_node = parent_node + delta_node_n
	     delta_node_n = (#tree + 1) - node_n
	     node_n = (#tree + 1)
	 end
	 
	 return tree
end












p = 0
x = {}
for k = 0,63,1 do x[k] = complex_rectan(p, p+1)  p = p + 2  end
tree = generate_tree(x)

leaf_code_ = {}
leaf_param_ = {}
fft(x, tree, 0, true, leaf_code_, leaf_param_)

for k=1,#leaf_code_,1 do print(leaf_code_[k]) end
for k=1,#leaf_param_,1 do print(leaf_param_[k]) end






--judge if the table is empty
--local next = next
--if next(my_table) == nil then
   --my_table is empty
--end