require('gnuplot')

function argmax(x)
	local index = 1
	local max = x[1]
	for i = 2, x:size(1) do
		if x[i] > max then
			index = i
			max = x[i]
		end
	end
	return index, max
end

function getOrderedKeys(t)
    local ordered = {}
    for k in pairs(t) do
        ordered[#ordered+1] = k
    end
    table.sort(ordered)
    return ordered
end

function print_read_max(model)
	local read_weights = model:get_read_weights()
	local num_heads = model.read_heads
	local fmt = '%-4d %.4f'
	if num_heads == 1 then
		printf(fmt .. '\n', argmax(read_weights))
	else
		local s = ''
		for i = 1, num_heads do
			s = s .. string.format(fmt, argmax(read_weights[i]))
			if i < num_heads then s = s .. ' | ' end
		end
		print(s)
	end
end

function print_write_max(model)
	local write_weights = model:get_write_weights()
	local num_heads = model.write_heads
	local fmt = '%-4d %.4f'
	if num_heads == 1 then
		printf(fmt .. '\n', argmax(write_weights))
	else
		local s = ''
		for i = 1, num_heads do
			s = s .. string.format(fmt, argmax(write_weights[i]))
			if i < num_heads then s = s .. ' | ' end
		end
		print(s)
	end
end

local function _save_plots_copy(ioExample, output, fileNamePrefix)
  gnuplot.pngfigure(string.format('logs/%s_target.png', fileNamePrefix))
  gnuplot.figure(1)
  gnuplot.raw('set cbrange [0:1]')
  gnuplot.imagesc(ioExample.target, 'color')

  gnuplot.pngfigure(string.format('logs/%s_output.png', fileNamePrefix))
  gnuplot.figure(2)
  gnuplot.raw('set cbrange [0:1]')
  gnuplot.imagesc(output, 'color')

  gnuplot.pngfigure('logs/error.png')
  gnuplot.figure(3)
  gnuplot.raw('set cbrange[-1:1]')
  gnuplot.imagesc(torch.csub(output, ioExample.target), 'color')

  gnuplot.plotflush(1)
  gnuplot.plotflush(2)
  gnuplot.plotflush(3)
end

local function _save_plots_recall(ioExample, output, fileNamePrefix)
    _save_plots_copy(ioExample, output, fileNamePrefix)
end

function save_plots(taskName, ioExample, output, lossHistory, fileNamePrefix)
  fileNamePrefix = fileNamePrefix or ''
  local savePltCmds = {
    copy = _save_plots_copy,
    recall = _save_plots_recall
  }
  savePltCmds[taskName](ioExample, output, fileNamePrefix)

  _plot_loss(lossHistory)
end

function _plot_loss(lossHistory)
  local xs, ys = {}, {}
  local orderedKeys = getOrderedKeys(lossHistory)
  print (orderedKeys)
  for i=1, #orderedKeys do
    xs[i] = orderedKeys[i]
    ys[i] = lossHistory[orderedKeys[i]]
  end
  gnuplot.pngfigure(string.format('logs/loss.png', fileNamePrefix))
  gnuplot.figure(4)
  gnuplot.plot(torch.Tensor(xs), torch.Tensor(ys))
  gnuplot.plotflush(4)
end