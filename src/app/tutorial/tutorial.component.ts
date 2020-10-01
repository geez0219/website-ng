import { Component, OnInit, HostListener, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { ActivatedRoute, Router, UrlSegment } from '@angular/router';
import { BehaviorSubject } from 'rxjs';
import { Title } from '@angular/platform-browser';

import { NestedTreeControl } from '@angular/cdk/tree';
import { MatSidenav } from '@angular/material/sidenav';
import { MatTreeNestedDataSource } from '@angular/material/tree';

import { Example } from '../example';
import { GlobalService } from '../global.service';

@Component({
  selector: 'app-tutorial',
  templateUrl: './tutorial.component.html',
  styleUrls: ['../api/api.component.css']
})
export class TutorialComponent implements OnInit, AfterViewChecked {
  tutorialList: Example[];
  selectedTutorial: string;
  currentSelection: string;
  currentTutorialText: string;
  scrollCounter = 0;
  scrollThreshold = 20;
  segments: UrlSegment[];
  fragment: string;
  currentVersion: string;

  treeControl: NestedTreeControl<Example>;
  dataSource: MatTreeNestedDataSource<Example>;
  minWidth = 640;
  screenWidth: number;
  private screenWidth$ = new BehaviorSubject<number>(window.innerWidth);

  subscribeTimer = -1;
  timeLeft = 10;
  structureHeaderDict = {
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/plain',
    'Access-Control-Allow-Origin': '*'
  }
  structureRequestOptions = {
    headers: new HttpHeaders(this.structureHeaderDict),
  };

  contentHeaderDict = {
    'Accept': 'application/json, text/plain',
    'Access-Control-Allow-Origin': '*'
  };

  contentRequestOptions = {
    responseType: 'text' as 'text',
    headers: new HttpHeaders(this.contentHeaderDict)
  };

  @ViewChild('sidenav', { static: true })
  sidenav: MatSidenav;

  @ViewChild('grippy', { static: true })
  grippy: ElementRef;

  @HostListener('window:resize', ['$event'])
  onResize(event) {
    this.screenWidth$.next(event.target.innerWidth);
  }

  @HostListener('window:scroll', ['$event'])
  onScroll(event) {
    if (this.scrollCounter < this.scrollThreshold) {
      this.scrollCounter += 1;
    }
  }

  constructor(private http: HttpClient,
              private router: Router,
              private route: ActivatedRoute,
              private title: Title,
              private globalService: GlobalService) { }

  ngOnInit() {
    this.treeControl = new NestedTreeControl<Example>(node => node.children);
    this.dataSource = new MatTreeNestedDataSource<Example>();

    this.route.url.subscribe((segments: UrlSegment[]) => {
      this.globalService.setLoading();
      this.segments = segments;

      this.currentVersion = this.segments[0].toString();
      this.getTutorialStructure();
    });

    this.globalService.version.subscribe((version: string) => {
      this.tutorialList = undefined;
    });

    this.route.fragment.subscribe((fragment: string) => {
      this.fragment = fragment;
    });

    this.screenWidth$.subscribe(width => {
      this.screenWidth = width;
    });
  }

  ngAfterViewChecked() {
    /* Scroll to the fragment postition.
       The reason for adding a scroll threshold is to scroll to the fragment position after image loaded.
       Before the counter goes to scollThreshold, the scrolling postion will stick to fragment.
       This wook will trigger the onScroll event and trigger back to this funtion again.
       So the counter will goes extremely fast.
    */

    if (this.fragment && this.scrollCounter < this.scrollThreshold) {
      if (document.querySelector('#' + this.fragment) != null) {
        document.querySelector('#' + this.fragment).scrollIntoView();
        window.scrollBy(0, -90); // the offset of navbar height
      }
    }
  }
  hasChild = (_: number, node: Example) => !!node.children && node.children.length > 0;

  flatten(arr) {
    let ret: Example[] = [];
    for (const a of arr) {
      if (a.children) {
        ret = ret.concat(this.flatten(a.children));
      } else {
        ret = ret.concat(a);
      }
    }

    return ret;
  }

  expandNodes(tutorialName: string) {
    const tutorialParts: Array<string> = tutorialName.split('/');
    tutorialParts.pop();

    const expandNode = this.tutorialList.filter(tutorial => tutorial.name === tutorialParts[0])[0];
    this.treeControl.expand(expandNode);
  }

  getTutorialStructure() {
    if (this.tutorialList) {
      this.loadSelectedTutorial();
    } else {
      this.http.get('assets/branches/' + this.currentVersion + '/tutorial/structure.json', this.structureRequestOptions).subscribe(data => {
        this.tutorialList = (data as Example[]);

        this.dataSource.data = this.tutorialList;
        this.treeControl.dataNodes = this.tutorialList;

        this.loadSelectedTutorial();
      },
        error => {
          console.error(error);
          this.globalService.resetLoading();
        });
    }
  }

  private loadSelectedTutorial() {
    if (this.segments.length === 0) {
      this.updateTutorialContent(this.tutorialList[0].children[0]);
      this.treeControl.expand(this.treeControl.dataNodes[0]);
    }
    else {
      const e: Example[] = this.flatten(this.tutorialList)
        .filter(tutorial =>
          (this.segments.map(segment => segment.toString()).slice(1, this.segments.length).join('/') + '.md') === tutorial.name);

      if (e.length > 0) {
        this.updateTutorialContent(e[0]);
        this.expandNodes(e[0].name);
      } else {
        this.globalService.resetLoading();
        this.router.navigate(['PageNotFound'], { replaceUrl: true });
      }
    }
  }

  private updateTutorialContent(tutorial: Example) {
    window.scroll(0, 0);

    this.selectedTutorial = tutorial.name;
    this.currentSelection = 'assets/branches/' + this.currentVersion + '/tutorial/' + tutorial.name;

    this.getSelectedTutorialText();
    this.title.setTitle(tutorial.displayName + ' | Fastestimator');
  }

  getSelectedTutorialText() {
    this.http.get(this.currentSelection, this.contentRequestOptions).subscribe(data => {
      this.currentTutorialText = data;
      this.globalService.resetLoading();
    },
      error => {
        console.error(error);
        this.globalService.resetLoading();
        this.router.navigate(['PageNotFound'], { replaceUrl: true });
      });
  }

  getImageUrl() {
    if (this.sidenav.opened) {
      this.grippy.nativeElement.style.left = '19rem';
      return 'url(../../assets/images/sidebar-grippy-hide.png)';
    } else {
      this.grippy.nativeElement.style.left = '0rem';
      return 'url(../../assets/images/sidebar-grippy-show.png)';
    }
  }

  checkSidebar() {
    if (this.sidenav.opened) {
      this.grippy.nativeElement.style.backgroundImage = 'url(../../assets/images/sidebar-grippy-hide.png)';
      this.grippy.nativeElement.style.left = '19rem';
    } else {
      this.grippy.nativeElement.style.backgroundImage = 'url(../../assets/images/sidebar-grippy-show.png)';
      this.grippy.nativeElement.style.left = '0rem';
    }
  }

  toggleMenu() {
    this.sidenav.toggle();
    this.checkSidebar();
  }

  createRouterLink(url: string) {
    const components: Array<string> = url.substring(0, url.length - 3).split('/');
    const ret = ['/tutorials', this.currentVersion];

    return ret.concat(components);
  }
}
