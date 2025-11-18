#!/bin/bash

# Qdrant Docker 安装脚本
# 用途：帮助用户手动安装和运行 Qdrant

set -e

echo "=========================================="
echo "Qdrant Docker 安装助手"
echo "=========================================="
echo ""

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ 错误: Docker 未运行，请先启动 Docker Desktop"
    exit 1
fi

echo "✅ Docker 正在运行"
echo ""

# 选择操作模式
echo "请选择操作模式："
echo "1) 从 Docker Hub 拉取并运行（需要网络）"
echo "2) 从本地 tar 文件加载镜像"
echo "3) 运行已存在的镜像"
echo "4) 保存镜像为 tar 文件"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "正在从 Docker Hub 拉取 Qdrant 镜像..."
        docker pull qdrant/qdrant:latest
        
        echo ""
        echo "正在启动 Qdrant 容器..."
        docker run -d --name qdrant \
            -p 6333:6333 \
            -p 6334:6334 \
            -v $(pwd)/qdrant_storage:/qdrant/storage \
            qdrant/qdrant
        
        echo "✅ Qdrant 已启动"
        echo "访问地址: http://localhost:6333/dashboard"
        ;;
    
    2)
        echo ""
        read -p "请输入 tar 文件路径: " tar_file
        
        if [ ! -f "$tar_file" ]; then
            echo "❌ 错误: 文件不存在: $tar_file"
            exit 1
        fi
        
        echo "正在加载镜像..."
        docker load -i "$tar_file"
        
        echo ""
        echo "正在启动 Qdrant 容器..."
        docker run -d --name qdrant \
            -p 6333:6333 \
            -p 6334:6334 \
            -v $(pwd)/qdrant_storage:/qdrant/storage \
            qdrant/qdrant
        
        echo "✅ Qdrant 已启动"
        ;;
    
    3)
        # 检查镜像是否存在
        if ! docker images | grep -q qdrant/qdrant; then
            echo "❌ 错误: 未找到 qdrant/qdrant 镜像"
            echo "请先选择选项 1 或 2 来加载镜像"
            exit 1
        fi
        
        # 检查容器是否已存在
        if docker ps -a | grep -q qdrant; then
            echo "检测到已存在的 qdrant 容器"
            read -p "是否启动现有容器? (y/n): " start_existing
            
            if [ "$start_existing" = "y" ]; then
                docker start qdrant
                echo "✅ Qdrant 容器已启动"
            else
                echo "正在创建新容器..."
                docker run -d --name qdrant \
                    -p 6333:6333 \
                    -p 6334:6334 \
                    -v $(pwd)/qdrant_storage:/qdrant/storage \
                    qdrant/qdrant
                echo "✅ Qdrant 容器已创建并启动"
            fi
        else
            echo "正在创建并启动 Qdrant 容器..."
            docker run -d --name qdrant \
                -p 6333:6333 \
                -p 6334:6334 \
                -v $(pwd)/qdrant_storage:/qdrant/storage \
                qdrant/qdrant
            echo "✅ Qdrant 容器已创建并启动"
        fi
        ;;
    
    4)
        # 检查镜像是否存在
        if ! docker images | grep -q qdrant/qdrant; then
            echo "❌ 错误: 未找到 qdrant/qdrant 镜像"
            echo "请先选择选项 1 来拉取镜像"
            exit 1
        fi
        
        read -p "请输入保存路径和文件名 (例如: ./qdrant-image.tar): " save_path
        
        echo "正在保存镜像..."
        docker save qdrant/qdrant:latest -o "$save_path"
        
        if [ -f "$save_path" ]; then
            file_size=$(du -h "$save_path" | cut -f1)
            echo "✅ 镜像已保存到: $save_path"
            echo "文件大小: $file_size"
        else
            echo "❌ 保存失败"
            exit 1
        fi
        ;;
    
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="

# 等待容器启动
sleep 2

# 检查容器状态
if docker ps | grep -q qdrant; then
    echo "✅ 容器运行中"
    echo ""
    echo "容器信息:"
    docker ps | grep qdrant
    echo ""
    
    # 检查健康状态
    echo "检查健康状态..."
    if curl -s http://localhost:6333/healthz > /dev/null; then
        echo "✅ Qdrant 健康检查通过"
    else
        echo "⚠️  健康检查失败，但容器正在运行"
        echo "   请稍等片刻后再次检查: curl http://localhost:6333/healthz"
    fi
else
    echo "⚠️  容器未运行，请检查日志: docker logs qdrant"
fi

echo ""
echo "=========================================="
echo "常用命令"
echo "=========================================="
echo "查看日志:    docker logs -f qdrant"
echo "停止容器:    docker stop qdrant"
echo "启动容器:    docker start qdrant"
echo "删除容器:    docker rm qdrant"
echo "访问面板:    http://localhost:6333/dashboard"
echo ""

