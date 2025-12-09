import { Heart } from 'lucide-react';

export function Footer() {
    return (
        <footer className="border-t border-white/5 bg-black z-10 relative">
            <div className="container mx-auto px-6 py-12">
                <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                    <div className="text-center md:text-left">
                        <h3 className="text-sm font-semibold text-slate-200">
                            Currency<span className="text-cyan-400">Guard</span> System
                        </h3>
                        <p className="text-xs text-slate-500 mt-2">
                            Advanced Fake Currency Detection using ResNet50 & Gemini Vision
                        </p>
                    </div>

                    <div className="flex items-center gap-2 text-sm text-slate-500">
                        <span>Built with</span>
                        <Heart className="w-4 h-4 text-red-500 fill-red-500/20" />
                        <span>by Computer Vision Team</span>
                    </div>

                    <div className="text-xs text-slate-600 text-center md:text-right">
                        <p>&copy; {new Date().getFullYear()} Fast Nuces. All rights reserved.</p>
                    </div>
                </div>
            </div>
        </footer>
    );
}
